#include "tokenizer.hpp"

#include <unicode/regex.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <unicode/utf16.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {

constexpr int NUM_RESERVED_SPECIAL_TOKENS = 256;
constexpr int BASE_VOCAB_SIZE = 128000;
constexpr char LLAMA_PRETOKENIZER_PATTERN[] =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

bool fail(std::string *error, const std::string &message) {
    if (error != nullptr) {
        *error = message;
    }
    return false;
}

std::string unicode_to_utf8(const icu::UnicodeString &text) {
    std::string result;
    text.toUTF8String(result);
    return result;
}

std::string base64_decode(const std::string &encoded) {
    constexpr char BASE64_ALPHABET[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    const std::string alphabet(BASE64_ALPHABET);
    std::string decoded;
    int buffer = 0;
    int bits_in_buffer = 0;

    for (char character : encoded) {
        if (character == '=') {
            break;
        }

        const std::size_t value = alphabet.find(character);
        if (value == std::string::npos) {
            throw std::runtime_error("Tokenizer file contains invalid Base64");
        }

        buffer = (buffer << 6) | static_cast<int>(value);
        bits_in_buffer += 6;
        if (bits_in_buffer >= 8) {
            bits_in_buffer -= 8;
            decoded.push_back(static_cast<char>((buffer >> bits_in_buffer) & 0xFF));
            buffer &= (1 << bits_in_buffer) - 1;
        }
    }
    return decoded;
}

}  // namespace

bool Tokenizer::load(const std::string &model_path, std::string *error) {
    std::ifstream tokenizer_file(model_path, std::ios::binary);
    if (!tokenizer_file.is_open()) {
        return fail(error, "Could not open tokenizer file: " + model_path);
    }

    token_to_id_.clear();
    id_to_token_.clear();
    special_token_to_id_.clear();
    id_to_special_token_.clear();

    std::vector<bool> seen_ids;
    std::string line;
    int line_number = 0;

    while (std::getline(tokenizer_file, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }

        std::istringstream line_stream(line);
        std::string encoded_bytes;
        std::string id_text;
        std::string extra_field;
        if (!(line_stream >> encoded_bytes >> id_text) || (line_stream >> extra_field)) {
            return fail(error, "Invalid tokenizer entry at line " + std::to_string(line_number));
        }

        int token_id = 0;
        std::string token_bytes;
        try {
            std::size_t parsed_characters = 0;
            token_id = std::stoi(id_text, &parsed_characters);
            if (parsed_characters != id_text.size()) {
                return fail(error, "Invalid token ID at line " + std::to_string(line_number));
            }
            token_bytes = base64_decode(encoded_bytes);
        } catch (const std::exception &exception) {
            return fail(
                error,
                "Invalid tokenizer entry at line " + std::to_string(line_number) + ": " + exception.what());
        }

        if (token_id < 0 || token_to_id_.find(token_bytes) != token_to_id_.end()) {
            return fail(error, "Duplicate token or invalid ID at line " + std::to_string(line_number));
        }
        if (static_cast<std::size_t>(token_id) >= id_to_token_.size()) {
            id_to_token_.resize(static_cast<std::size_t>(token_id) + 1);
            seen_ids.resize(static_cast<std::size_t>(token_id) + 1, false);
        }
        if (seen_ids[token_id]) {
            return fail(error, "Duplicate token ID at line " + std::to_string(line_number));
        }

        token_to_id_.emplace(token_bytes, token_id);
        id_to_token_[token_id] = std::move(token_bytes);
        seen_ids[token_id] = true;
    }

    if (token_to_id_.size() != BASE_VOCAB_SIZE || id_to_token_.size() != BASE_VOCAB_SIZE ||
        std::find(seen_ids.begin(), seen_ids.end(), false) != seen_ids.end()) {
        return fail(error, "Expected exactly 128000 contiguous base Llama token IDs");
    }

    std::vector<std::string> special_tokens = {
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",
    };
    for (int i = 5; i < NUM_RESERVED_SPECIAL_TOKENS - 5; ++i) {
        special_tokens.push_back("<|reserved_special_token_" + std::to_string(i) + "|>");
    }

    if (special_tokens.size() != NUM_RESERVED_SPECIAL_TOKENS) {
        return fail(error, "Internal error constructing Llama special tokens");
    }

    id_to_special_token_ = special_tokens;
    for (std::size_t i = 0; i < special_tokens.size(); ++i) {
        special_token_to_id_.emplace(special_tokens[i], BASE_VOCAB_SIZE + static_cast<int>(i));
    }
    return true;
}

std::vector<std::string> Tokenizer::pretokenize(const std::string &text) {
    UErrorCode status = U_ZERO_ERROR;
    const icu::UnicodeString pattern = icu::UnicodeString::fromUTF8(LLAMA_PRETOKENIZER_PATTERN);
    std::unique_ptr<icu::RegexPattern> regex(icu::RegexPattern::compile(pattern, 0, status));
    if (U_FAILURE(status)) {
        throw std::runtime_error("Could not compile the Llama pre-tokenization regex");
    }

    const icu::UnicodeString unicode_text = icu::UnicodeString::fromUTF8(text);
    std::unique_ptr<icu::RegexMatcher> matcher(regex->matcher(unicode_text, status));
    if (U_FAILURE(status)) {
        throw std::runtime_error("Could not create the Llama pre-tokenization matcher");
    }

    std::vector<std::string> pieces;
    while (matcher->find(status)) {
        const icu::UnicodeString piece = matcher->group(status);
        if (U_FAILURE(status)) {
            throw std::runtime_error("Llama pre-tokenization failed");
        }
        pieces.push_back(unicode_to_utf8(piece));
    }
    if (U_FAILURE(status)) {
        throw std::runtime_error("Llama pre-tokenization failed");
    }
    return pieces;
}

std::vector<int> Tokenizer::encode_piece(const std::string &piece) const {
    std::vector<std::string> parts;
    parts.reserve(piece.size());
    for (unsigned char byte : piece) {
        parts.emplace_back(1, static_cast<char>(byte));
    }

    // BPE repeatedly merges the adjacent pair whose learned rank is smallest.
    while (parts.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        std::size_t best_index = parts.size();

        for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
            const auto candidate = token_to_id_.find(parts[i] + parts[i + 1]);
            if (candidate != token_to_id_.end() && candidate->second < best_rank) {
                best_rank = candidate->second;
                best_index = i;
            }
        }
        if (best_index == parts.size()) {
            break;
        }

        parts[best_index] += parts[best_index + 1];
        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
    }

    std::vector<int> token_ids;
    token_ids.reserve(parts.size());
    for (const std::string &part : parts) {
        token_ids.push_back(token_to_id_.at(part));
    }
    return token_ids;
}

std::vector<int> Tokenizer::encode(const std::string &text, bool add_bos, bool add_eos) const {
    if (token_to_id_.empty()) {
        throw std::runtime_error("Load the tokenizer before encoding text");
    }

    std::vector<int> token_ids;
    for (const std::string &piece : pretokenize(text)) {
        const std::vector<int> piece_ids = encode_piece(piece);
        token_ids.insert(token_ids.end(), piece_ids.begin(), piece_ids.end());
    }
    if (add_bos) {
        token_ids.insert(token_ids.begin(), bos_id());
    }
    if (add_eos) {
        token_ids.push_back(eos_id());
    }
    return token_ids;
}

std::string Tokenizer::decode(const std::vector<int> &token_ids) const {
    if (token_to_id_.empty()) {
        throw std::runtime_error("Load the tokenizer before decoding IDs");
    }

    std::string text;
    for (int token_id : token_ids) {
        if (token_id >= 0 && static_cast<std::size_t>(token_id) < id_to_token_.size()) {
            text += id_to_token_[token_id];
            continue;
        }

        const int special_index = token_id - BASE_VOCAB_SIZE;
        if (special_index >= 0 && static_cast<std::size_t>(special_index) < id_to_special_token_.size()) {
            text += id_to_special_token_[special_index];
            continue;
        }
        throw std::runtime_error("Token ID is outside the Llama 3.2 vocabulary: " + std::to_string(token_id));
    }
    return text;
}

std::vector<int> Tokenizer::encode_header(const ChatMessage &message) const {
    std::vector<int> token_ids = {special_token_id("<|start_header_id|>")};
    const std::vector<int> role_ids = encode(message.role, false, false);
    token_ids.insert(token_ids.end(), role_ids.begin(), role_ids.end());
    token_ids.push_back(special_token_id("<|end_header_id|>"));

    const std::vector<int> newline_ids = encode("\n\n", false, false);
    token_ids.insert(token_ids.end(), newline_ids.begin(), newline_ids.end());
    return token_ids;
}

std::vector<int> Tokenizer::encode_dialog_prompt(const std::vector<ChatMessage> &dialog) const {
    std::vector<int> token_ids = {bos_id()};
    for (const ChatMessage &message : dialog) {
        const std::vector<int> header_ids = encode_header(message);
        token_ids.insert(token_ids.end(), header_ids.begin(), header_ids.end());

        const std::vector<int> content_ids = encode(trim_unicode_whitespace(message.content), false, false);
        token_ids.insert(token_ids.end(), content_ids.begin(), content_ids.end());
        token_ids.push_back(eot_id());
    }

    const std::vector<int> assistant_header = encode_header({"assistant", ""});
    token_ids.insert(token_ids.end(), assistant_header.begin(), assistant_header.end());
    return token_ids;
}

std::string Tokenizer::trim_unicode_whitespace(const std::string &text) const {
    const icu::UnicodeString unicode_text = icu::UnicodeString::fromUTF8(text);
    int32_t start = 0;
    int32_t end = unicode_text.length();

    while (start < end) {
        const UChar32 character = unicode_text.char32At(start);
        if (!u_isUWhiteSpace(character)) {
            break;
        }
        start += U16_LENGTH(character);
    }
    while (end > start) {
        const UChar32 character = unicode_text.char32At(end - 1);
        const int32_t character_start = end - U16_LENGTH(character);
        if (!u_isUWhiteSpace(character)) {
            break;
        }
        end = character_start;
    }
    return unicode_to_utf8(unicode_text.tempSubStringBetween(start, end));
}

int Tokenizer::special_token_id(const std::string &token) const {
    const auto found = special_token_to_id_.find(token);
    if (found == special_token_to_id_.end()) {
        throw std::runtime_error("Unknown Llama special token: " + token);
    }
    return found->second;
}

std::size_t Tokenizer::base_vocab_size() const {
    return token_to_id_.size();
}

std::size_t Tokenizer::vocab_size() const {
    return token_to_id_.size() + special_token_to_id_.size();
}

int Tokenizer::bos_id() const {
    return special_token_id("<|begin_of_text|>");
}

int Tokenizer::eos_id() const {
    return special_token_id("<|end_of_text|>");
}

int Tokenizer::eot_id() const {
    return special_token_id("<|eot_id|>");
}
