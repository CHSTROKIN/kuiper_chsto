// =============================================================================
// chat.cpp  —  Chat with a Llama 3.x Instruct model in the terminal.
//
// A conversation is one long token sequence in the Llama 3 chat format:
//
//   <|begin_of_text|>
//   <|start_header_id|>system<|end_header_id|>\n\n ... <|eot_id|>
//   <|start_header_id|>user<|end_header_id|>\n\n ... <|eot_id|>
//   <|start_header_id|>assistant<|end_header_id|>\n\n  <- the model continues here
//
// and it ends its answer with <|eot_id|>. The KV cache remembers everything
// that was already processed, so each turn only feeds the new tokens.
//
// Usage: chat [--model-dir DIR] [--device cuda|cpu] [--temperature 0.6]
//             [--top-p 0.9] [--max-tokens 512] [--ctx 4096] [--seed N]
//             [--system TEXT] [--prompt TEXT] [--stats]
//
//   --temperature 0   greedy decoding (always the most likely token)
//   --prompt          answer one message and exit
//
// While chatting:  /reset starts a new conversation,  /exit quits.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "app_utils.h"
#include "llama3/checkpoint.h"
#include "llama3/model.h"
#include "llama3/tokenizer.h"

namespace {

// ---- picking the next token ----------------------------------------------
// temperature rescales the logits before softmax (< 1 = more predictable).
// top-p ("nucleus") sampling then keeps only the most likely tokens whose
// probabilities add up to top_p, and draws one of them at random.
class Sampler {
 public:
  Sampler(float temperature, float top_p, uint64_t seed)
      : temperature_(temperature), top_p_(top_p), rng_(seed) {}

  int sample(const float* logits, int n) {
    const int best = llama3::cpu::argmax(logits, n);
    if (temperature_ <= 0) return best;

    probs_.resize(n);
    for (int i = 0; i < n; ++i) probs_[i] = (logits[i] - logits[best]) / temperature_;
    double sum = 0;
    for (float& p : probs_) {
      p = std::exp(p);
      sum += p;
    }

    // Tokens below this are far too unlikely to matter; skipping them keeps the
    // sort small.
    const float floor = static_cast<float>(sum) * 1e-7f;
    candidates_.clear();
    for (int i = 0; i < n; ++i) {
      if (probs_[i] > floor) candidates_.push_back(i);
    }
    std::sort(candidates_.begin(), candidates_.end(),
              [&](int a, int b) { return probs_[a] > probs_[b]; });

    size_t kept = 0;
    double mass = 0;
    while (kept < candidates_.size() && mass < top_p_ * sum) mass += probs_[candidates_[kept++]];
    if (kept == 0) return best;

    double r = std::uniform_real_distribution<double>(0, mass)(rng_);
    for (size_t i = 0; i < kept; ++i) {
      r -= probs_[candidates_[i]];
      if (r <= 0) return candidates_[i];
    }
    return candidates_[kept - 1];
  }

 private:
  float temperature_, top_p_;
  std::mt19937_64 rng_;
  std::vector<float> probs_;
  std::vector<int> candidates_;
};

// ---- one interface for both devices --------------------------------------
class Engine {
 public:
  virtual ~Engine() = default;
  // Feeds one token. When need_logits is true, returns the logits (on the CPU).
  virtual const float* feed(int token, int pos, bool need_logits) = 0;
};

class CpuEngine : public Engine {
 public:
  CpuEngine(const llama3::ModelConfig& c, llama3::Weights weights)
      : weights_(std::move(weights)), model_(c, weights_) {}
  const float* feed(int token, int pos, bool need_logits) override {
    return model_.forward(token, pos, need_logits);
  }

 private:
  llama3::Weights weights_;  // must be declared before model_, which refers to it
  llama3::CpuLlama3 model_;
};

class CudaEngine : public Engine {
 public:
  CudaEngine(const llama3::ModelConfig& c, const llama3::Weights& weights) : model_(c, weights) {}
  const float* feed(int token, int pos, bool need_logits) override {
    model_.forward(token, pos, need_logits);
    if (!need_logits) {
      // wait, so the launch queue stays short and timings are honest
      CUDA_CHECK(cudaDeviceSynchronize());
      return nullptr;
    }
    logits_ = model_.logits_on_cpu();
    return logits_.data();
  }

 private:
  llama3::CudaLlama3 model_;
  std::vector<float> logits_;
};

// ---- printing --------------------------------------------------------------
// A token can end in the middle of a multi-byte character (common in Chinese),
// so only complete UTF-8 characters are printed; the rest waits.
size_t complete_utf8_length(const std::string& s) {
  size_t start = s.size();
  // walk back over at most 3 continuation bytes to the last lead byte
  for (int back = 1; back <= 4 && start > 0; ++back) {
    const unsigned char c = static_cast<unsigned char>(s[--start]);
    if ((c & 0xC0) == 0x80) continue;
    const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    return start + need <= s.size() ? s.size() : start;
  }
  return s.size();
}

std::string today() {
  const std::time_t now = std::time(nullptr);
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%d %b %Y", std::localtime(&now));
  return buffer;
}

// ---- the conversation --------------------------------------------------------
class Chat {
 public:
  Chat(const llama3::Tokenizer& tokenizer, Engine& engine, Sampler& sampler,
       const llama3::ModelConfig& c, std::string system, int max_tokens, bool stats)
      : tok_(tokenizer), engine_(engine), sampler_(sampler), config_(c),
        system_(std::move(system)), max_tokens_(max_tokens), stats_(stats),
        eot_(tokenizer.special("<|eot_id|>")) {
    reset();
  }

  void reset() {
    pos_ = 0;
    pending_.clear();
    pending_.push_back(tok_.special("<|begin_of_text|>"));
    std::string text = "Cutting Knowledge Date: December 2023\nToday Date: " + today() + "\n\n";
    add_message("system", text + system_);
  }

  // Adds the user's message and streams the answer to stdout. Returns false if
  // the conversation no longer fits in the context.
  bool reply(const std::string& message) {
    const size_t before = pending_.size();
    add_message("user", message);
    add_header("assistant");

    // every pending token plus at least one answer token must fit
    if (pos_ + static_cast<int>(pending_.size()) >= config_.max_seq_len) {
      pending_.resize(before);
      return false;
    }

    // 1. read the prompt: only the last token needs logits
    app::CpuTimer timer;
    timer.start();
    const int prompt_tokens = static_cast<int>(pending_.size());
    const float* logits = nullptr;
    for (size_t i = 0; i < pending_.size(); ++i) {
      logits = engine_.feed(pending_[i], pos_++, i + 1 == pending_.size());
    }
    pending_.clear();
    const double prompt_ms = timer.elapsed_ms();

    // 2. generate one token at a time until the model ends its turn
    timer.start();
    std::string unprinted;
    int generated = 0;
    bool finished = false;
    while (true) {
      const int next = sampler_.sample(logits, config_.vocab_size);
      ++generated;
      if (is_end(next)) {
        finished = true;
        break;
      }
      unprinted += tok_.token_bytes(next);
      const size_t ready = complete_utf8_length(unprinted);
      std::fwrite(unprinted.data(), 1, ready, stdout);
      std::fflush(stdout);
      unprinted.erase(0, ready);

      if (generated >= max_tokens_ || pos_ >= config_.max_seq_len) {
        pending_.push_back(next);  // printed, but not in the KV cache yet
        break;
      }
      logits = engine_.feed(next, pos_++, true);
    }
    std::fwrite(unprinted.data(), 1, unprinted.size(), stdout);
    if (!finished) std::printf(" [answer cut off]");
    std::printf("\n");
    std::fflush(stdout);  // keep it ahead of the stats on stderr

    // The end-of-turn token belongs to the history; feed it with the next turn.
    pending_.push_back(eot_);

    if (stats_) {
      std::fprintf(stderr, "[prompt %d tokens, %.0f ms | answer %d tokens, %.1f tokens/s | "
                           "context %d/%d]\n",
                   prompt_tokens, prompt_ms, generated,
                   1000.0 * generated / std::max(timer.elapsed_ms(), 1e-3), pos_,
                   config_.max_seq_len);
    }
    return true;
  }

 private:
  bool is_end(int token) const {
    // <|end_of_text|>, <|eom_id|>, <|eot_id|>
    return token == eot_ || token == tok_.special("<|end_of_text|>") ||
           token == tok_.special("<|eom_id|>");
  }

  void add_header(const std::string& role) {
    pending_.push_back(tok_.special("<|start_header_id|>"));
    append(tok_.encode(role));
    pending_.push_back(tok_.special("<|end_header_id|>"));
    append(tok_.encode("\n\n"));
  }

  void add_message(const std::string& role, const std::string& content) {
    add_header(role);
    append(tok_.encode(content));
    pending_.push_back(eot_);
  }

  void append(const std::vector<int>& ids) {
    pending_.insert(pending_.end(), ids.begin(), ids.end());
  }

  const llama3::Tokenizer& tok_;
  Engine& engine_;
  Sampler& sampler_;
  const llama3::ModelConfig& config_;
  const std::string system_;
  const int max_tokens_;
  const bool stats_;
  const int eot_;
  int pos_ = 0;               // number of tokens in the KV cache
  std::vector<int> pending_;  // tokens of the history that are not cached yet
};

int run(int argc, char** argv) {
  app::configure_cpu_threads();
  const std::string dir =
      app::get_arg(argc, argv, "--model-dir", "models/llama-3.2-1b-instruct");
  const std::string device = app::get_arg(argc, argv, "--device", "cuda");
  if (device != "cuda" && device != "cpu") {
    std::fprintf(stderr, "unknown --device '%s' (use cuda or cpu)\n", device.c_str());
    return 1;
  }

  llama3::ModelConfig c = llama3::load_config(dir);
  const int ctx = app::get_int_arg(argc, argv, "--ctx", 4096);
  if (ctx < 64) {
    std::fprintf(stderr, "--ctx must be at least 64\n");
    return 1;
  }
  c.max_seq_len = std::min(c.max_seq_len, ctx);

  if (device == "cuda") {
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const double cache_bytes = 8.0 * c.n_layers * c.max_seq_len * c.kv_dim();  // K + V, fp32
    const double need = llama3::weight_bytes(c) + cache_bytes;
    if (need > 0.95 * free_bytes) {
      std::fprintf(stderr, "Not enough GPU memory: need %.1f GB, %.1f GB free. "
                           "Use --device cpu or a smaller model.\n",
                   need / 1e9, free_bytes / 1e9);
      return 1;
    }
  }

  std::fprintf(stderr, "Loading %s (%.1f GB of fp32 weights, %s) ...\n", dir.c_str(),
               llama3::weight_bytes(c) / 1e9, device.c_str());
  app::CpuTimer timer;
  timer.start();
  const llama3::Tokenizer tokenizer(dir + "/tokenizer.json");
  std::unique_ptr<Engine> engine;
  {
    llama3::Weights weights = llama3::load_weights(dir, c);
    if (device == "cpu") {
      engine = std::make_unique<CpuEngine>(c, std::move(weights));
    } else {
      engine = std::make_unique<CudaEngine>(c, weights);  // the CPU copy is freed here
    }
  }
  std::fprintf(stderr, "Ready in %.1f s.\n", timer.elapsed_ms() / 1000);

  const float temperature =
      std::strtof(app::get_arg(argc, argv, "--temperature", "0.6").c_str(), nullptr);
  const float top_p = std::strtof(app::get_arg(argc, argv, "--top-p", "0.9").c_str(), nullptr);
  const std::string seed_arg = app::get_arg(argc, argv, "--seed", "");
  const uint64_t seed = seed_arg.empty() ? std::random_device{}()
                                         : std::strtoull(seed_arg.c_str(), nullptr, 10);
  Sampler sampler(temperature, top_p, seed);

  Chat chat(tokenizer, *engine, sampler, c,
            app::get_arg(argc, argv, "--system", "You are a helpful assistant."),
            std::max(1, app::get_int_arg(argc, argv, "--max-tokens", 512)),
            app::has_flag(argc, argv, "--stats"));

  const std::string prompt = app::get_arg(argc, argv, "--prompt", "");
  if (!prompt.empty()) {
    if (chat.reply(prompt)) return 0;
    std::fprintf(stderr, "The prompt does not fit in the context (--ctx %d).\n", c.max_seq_len);
    return 1;
  }

  std::fprintf(stderr, "Type a message. /reset = new conversation, /exit = quit.\n");
  std::string line;
  while (true) {
    std::printf("\nYou> ");
    std::fflush(stdout);
    if (!std::getline(std::cin, line) || line == "/exit") break;
    if (line.empty()) continue;
    if (line == "/reset") {
      chat.reset();
      std::printf("(new conversation)\n");
      continue;
    }
    std::printf("Llama> ");
    std::fflush(stdout);
    if (!chat.reply(line)) {
      std::printf("\n(The context is full. Type /reset to start a new conversation.)\n");
    }
  }
  std::printf("\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
