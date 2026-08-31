#pragma once

#include <string>
#include <unordered_map>
#include <vector>

// Small Llama 3.2 tokenizer for turning normal text into the IDs that the
// inference engine passes to embeddingGather.
class Tokenizer
{
public:
    bool load(const std::string &model_path, std::string *error = nullptr);

    std::vector<int> encode(const std::string &text, bool add_bos = true, bool add_eos = false) const;
    std::string decode(const std::vector<int> &token_ids) const;

    std::size_t base_vocab_size() const;

private:
    std::vector<std::string> split_text(const std::string &text) const;
    std::vector<int> encode_piece(const std::string &piece) const;

    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
};
