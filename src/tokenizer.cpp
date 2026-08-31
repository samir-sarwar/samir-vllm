#include "tokenizer.hpp"

#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace
{

    constexpr int BOS_ID = 128000;
    constexpr int EOS_ID = 128001;

    bool is_ascii_letter(unsigned char character)
    {
        return (character >= 'a' && character <= 'z') ||
               (character >= 'A' && character <= 'Z');
    }

    bool is_ascii_digit(unsigned char character)
    {
        return character >= '0' && character <= '9';
    }

    bool is_whitespace(unsigned char character)
    {
        return character == ' ' || character == '\t' || character == '\n' ||
               character == '\r' || character == '\f' || character == '\v';
    }

    bool is_contraction_at(const std::string &text, std::size_t position, std::size_t *length)
    {
        if (text[position] != '\'')
        {
            return false;
        }

        const std::string remaining = text.substr(position);
        const std::vector<std::string> contractions = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
        for (const std::string &contraction : contractions)
        {
            if (remaining.size() < contraction.size())
            {
                continue;
            }

            bool matches = true;
            for (std::size_t i = 0; i < contraction.size(); ++i)
            {
                if (std::tolower(static_cast<unsigned char>(remaining[i])) != contraction[i])
                {
                    matches = false;
                    break;
                }
            }
            if (matches)
            {
                *length = contraction.size();
                return true;
            }
        }
        return false;
    }

    std::string base64_decode(const std::string &encoded)
    {
        constexpr char BASE64_ALPHABET[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        std::string decoded;
        const std::string alphabet(BASE64_ALPHABET);
        int buffer = 0;
        int bits_in_buffer = 0;

        for (char character : encoded)
        {
            if (character == '=')
            {
                break;
            }
            const std::size_t value = alphabet.find(character);
            if (value == std::string::npos)
            {
                throw std::runtime_error("Tokenizer file contains invalid Base64");
            }

            buffer = (buffer << 6) | static_cast<int>(value);
            bits_in_buffer += 6;
            if (bits_in_buffer >= 8)
            {
                bits_in_buffer -= 8;
                decoded.push_back(static_cast<char>((buffer >> bits_in_buffer) & 0xFF));
                // Keep only the unfinished bits so buffer stays small even when
                // a vocabulary entry contains many bytes.
                buffer &= (1 << bits_in_buffer) - 1;
            }
        }
        return decoded;
    }

} // namespace

bool Tokenizer::load(const std::string &model_path, std::string *error)
{
    std::ifstream tokenizer_file(model_path, std::ios::binary);
    if (!tokenizer_file.is_open())
    {
        if (error != nullptr)
        {
            *error = "Could not open tokenizer file: " + model_path;
        }
        return false;
    }

    token_to_id_.clear();
    id_to_token_.clear();

    std::string line;
    while (std::getline(tokenizer_file, line))
    {
        std::istringstream line_stream(line);
        std::string encoded_bytes;
        int token_id = 0;
        if (!(line_stream >> encoded_bytes >> token_id))
        {
            if (error != nullptr)
            {
                *error = "Could not read one tokenizer entry";
            }
            return false;
        }

        std::string token_bytes;
        try
        {
            token_bytes = base64_decode(encoded_bytes);
        }
        catch (const std::exception &exception)
        {
            if (error != nullptr)
            {
                *error = exception.what();
            }
            return false;
        }

        if (token_id < 0 || token_to_id_.count(token_bytes) != 0)
        {
            if (error != nullptr)
            {
                *error = "Tokenizer file has a duplicate token or invalid ID";
            }
            return false;
        }
        if (static_cast<std::size_t>(token_id) >= id_to_token_.size())
        {
            id_to_token_.resize(static_cast<std::size_t>(token_id) + 1);
        }
        if (!id_to_token_[token_id].empty())
        {
            if (error != nullptr)
            {
                *error = "Tokenizer file has duplicate IDs";
            }
            return false;
        }

        token_to_id_[token_bytes] = token_id;
        id_to_token_[token_id] = std::move(token_bytes);
    }

    if (token_to_id_.size() != 128000 || id_to_token_.size() != 128000)
    {
        if (error != nullptr)
        {
            *error = "Expected 128000 base Llama tokens, loaded " + std::to_string(token_to_id_.size());
        }
        return false;
    }
    return true;
}

std::vector<std::string> Tokenizer::split_text(const std::string &text) const
{
    std::vector<std::string> pieces;
    std::size_t position = 0;

    while (position < text.size())
    {
        const unsigned char current = static_cast<unsigned char>(text[position]);
        std::size_t length = 0;

        // Llama's pre-tokenizer recognizes contractions before normal words.
        if (is_contraction_at(text, position, &length))
        {
            pieces.push_back(text.substr(position, length));
            position += length;
            continue;
        }

        // A normal word can include one leading space or punctuation character.
        const bool has_leading_character =
            current != '\r' && current != '\n' && !is_ascii_letter(current) && !is_ascii_digit(current) &&
            position + 1 < text.size() && is_ascii_letter(static_cast<unsigned char>(text[position + 1]));
        const std::size_t word_start = has_leading_character ? position + 1 : position;
        if (word_start < text.size() && is_ascii_letter(static_cast<unsigned char>(text[word_start])))
        {
            std::size_t end = word_start;
            while (end < text.size() && is_ascii_letter(static_cast<unsigned char>(text[end])))
            {
                ++end;
            }
            pieces.push_back(text.substr(position, end - position));
            position = end;
            continue;
        }

        // Llama's pattern keeps numbers in groups of at most three digits.
        if (is_ascii_digit(current))
        {
            std::size_t end = position;
            while (end < text.size() && end - position < 3 &&
                   is_ascii_digit(static_cast<unsigned char>(text[end])))
            {
                ++end;
            }
            pieces.push_back(text.substr(position, end - position));
            position = end;
            continue;
        }

        // Punctuation becomes a piece, optionally keeping one leading space.
        const bool has_leading_space = current == ' ' && position + 1 < text.size() &&
                                       !is_whitespace(static_cast<unsigned char>(text[position + 1])) &&
                                       !is_ascii_letter(static_cast<unsigned char>(text[position + 1])) &&
                                       !is_ascii_digit(static_cast<unsigned char>(text[position + 1]));
        const std::size_t punctuation_start = has_leading_space ? position + 1 : position;
        if (punctuation_start < text.size() && !is_whitespace(static_cast<unsigned char>(text[punctuation_start])) &&
            !is_ascii_letter(static_cast<unsigned char>(text[punctuation_start])) &&
            !is_ascii_digit(static_cast<unsigned char>(text[punctuation_start])))
        {
            std::size_t end = punctuation_start;
            while (end < text.size() && !is_whitespace(static_cast<unsigned char>(text[end])) &&
                   !is_ascii_letter(static_cast<unsigned char>(text[end])) &&
                   !is_ascii_digit(static_cast<unsigned char>(text[end])))
            {
                ++end;
            }
            while (end < text.size() && (text[end] == '\r' || text[end] == '\n'))
            {
                ++end;
            }
            pieces.push_back(text.substr(position, end - position));
            position = end;
            continue;
        }

        // Preserve whitespace as its own tokenization piece. For a run of spaces
        // before a word, leave one space behind so that it can join that word.
        std::size_t end = position;
        while (end < text.size() && is_whitespace(static_cast<unsigned char>(text[end])))
        {
            ++end;
        }
        if (end > position + 1 && end < text.size())
        {
            --end;
        }
        pieces.push_back(text.substr(position, end - position));
        position = end;
    }

    return pieces;
}

std::vector<int> Tokenizer::encode_piece(const std::string &piece) const
{
    std::vector<std::string> parts;
    for (unsigned char byte : piece)
    {
        parts.emplace_back(1, static_cast<char>(byte));
    }

    // BPE repeatedly merges the adjacent pair with the lowest learned rank.
    while (parts.size() > 1)
    {
        int best_rank = std::numeric_limits<int>::max();
        std::size_t best_index = parts.size();
        for (std::size_t i = 0; i + 1 < parts.size(); ++i)
        {
            const auto candidate = token_to_id_.find(parts[i] + parts[i + 1]);
            if (candidate != token_to_id_.end() && candidate->second < best_rank)
            {
                best_rank = candidate->second;
                best_index = i;
            }
        }
        if (best_index == parts.size())
        {
            break;
        }
        parts[best_index] += parts[best_index + 1];
        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
    }

    std::vector<int> token_ids;
    for (const std::string &part : parts)
    {
        token_ids.push_back(token_to_id_.at(part));
    }
    return token_ids;
}

std::vector<int> Tokenizer::encode(const std::string &text, bool add_bos, bool add_eos) const
{
    if (token_to_id_.empty())
    {
        throw std::runtime_error("Load the tokenizer before encoding text");
    }

    std::vector<int> token_ids;
    for (const std::string &piece : split_text(text))
    {
        const std::vector<int> piece_ids = encode_piece(piece);
        token_ids.insert(token_ids.end(), piece_ids.begin(), piece_ids.end());
    }
    if (add_bos)
    {
        token_ids.insert(token_ids.begin(), BOS_ID);
    }
    if (add_eos)
    {
        token_ids.push_back(EOS_ID);
    }
    return token_ids;
}

std::string Tokenizer::decode(const std::vector<int> &token_ids) const
{
    std::string text;
    for (int token_id : token_ids)
    {
        if (token_id == BOS_ID)
        {
            text += "<|begin_of_text|>";
        }
        else if (token_id == EOS_ID)
        {
            text += "<|end_of_text|>";
        }
        else if (token_id >= 0 && static_cast<std::size_t>(token_id) < id_to_token_.size())
        {
            text += id_to_token_[token_id];
        }
        else
        {
            throw std::runtime_error("Cannot decode an ID outside this basic tokenizer");
        }
    }
    return text;
}

std::size_t Tokenizer::base_vocab_size() const
{
    return token_to_id_.size();
}
