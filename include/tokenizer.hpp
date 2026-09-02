#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

struct ChatMessage {
    std::string role;
    std::string content;
};

// Llama 3.2's TikToken-based tokenizer. It uses the official tokenizer.model
// BPE ranks and reproduces Meta's Llama Instruct prompt format.
class Tokenizer {
public:
    bool load(const std::string &model_path, std::string *error = nullptr);

    // By default this behaves like Hugging Face's Llama tokenizer and prepends BOS.
    std::vector<int> encode(const std::string &text, bool add_bos = true, bool add_eos = false) const;
    std::string decode(const std::vector<int> &token_ids) const;
    std::vector<int> encode_dialog_prompt(const std::vector<ChatMessage> &dialog) const;

    std::size_t base_vocab_size() const;
    std::size_t vocab_size() const;
    int bos_id() const;
    int eos_id() const;
    int eot_id() const;

private:
    static std::vector<std::string> pretokenize(const std::string &text);
    std::vector<int> encode_piece(const std::string &piece) const;
    std::vector<int> encode_header(const ChatMessage &message) const;
    std::string trim_unicode_whitespace(const std::string &text) const;
    int special_token_id(const std::string &token) const;

    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> special_token_to_id_;
    std::vector<std::string> id_to_special_token_;
};
