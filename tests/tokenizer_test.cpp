#include "tokenizer.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void expect_ids(const std::vector<int> &actual, const std::vector<int> &expected, const std::string &name) {
    if (actual == expected) {
        return;
    }

    std::cerr << name << " failed\nExpected: ";
    for (int id : expected) {
        std::cerr << id << ' ';
    }
    std::cerr << "\nActual:   ";
    for (int id : actual) {
        std::cerr << id << ' ';
    }
    std::cerr << '\n';
    std::exit(1);
}

void expect_text(const std::string &actual, const std::string &expected, const std::string &name) {
    if (actual != expected) {
        std::cerr << name << " failed\nExpected: " << expected << "\nActual:   " << actual << '\n';
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: tokenizer-test path/to/tokenizer.model\n";
        return 1;
    }

    Tokenizer tokenizer;
    std::string error;
    if (!tokenizer.load(argv[1], &error)) {
        std::cerr << error << '\n';
        return 1;
    }

    if (tokenizer.base_vocab_size() != 128000 || tokenizer.vocab_size() != 128256) {
        std::cerr << "Unexpected Llama vocabulary size\n";
        return 1;
    }

    const std::vector<int> expected_sentence = {128000, 2028, 374, 264, 1296, 11914, 13, 128001};
    expect_ids(tokenizer.encode("This is a test sentence.", true, true), expected_sentence, "Meta basic encode");
    expect_text(
        tokenizer.decode(expected_sentence),
        "<|begin_of_text|>This is a test sentence.<|end_of_text|>",
        "Meta basic decode");
    expect_ids(tokenizer.encode("Hello, world!", false), {9906, 11, 1917, 0}, "punctuation");
    expect_ids(
        tokenizer.encode("We're testing 123456 and don't stop.", false),
        {1687, 2351, 7649, 220, 4513, 10961, 323, 1541, 956, 3009, 13},
        "contractions and numbers");
    expect_ids(
        tokenizer.encode("  spaces\tand\nnewlines  ", false),
        {220, 12908, 53577, 198, 943, 8128, 256},
        "whitespace");
    expect_ids(tokenizer.encode("Héllo 😊\n世界", false), {39, 19010, 385, 27623, 232, 198, 102616}, "Unicode");
    expect_ids(
        tokenizer.encode("<|begin_of_text|> is ordinary input by default", false),
        {27, 91, 7413, 3659, 4424, 91, 29, 374, 19664, 1988, 555, 1670},
        "special-token-looking text");

    expect_ids(
        tokenizer.encode_dialog_prompt({
            {"system", "Be concise."},
            {"user", "  Explain UTF-8.  "},
        }),
        {128000, 128006, 9125, 128007, 271, 3513, 64694, 13, 128009, 128006, 882, 128007, 271,
         849, 21435, 20677, 12, 23, 13, 128009, 128006, 78191, 128007, 271},
        "Meta Instruct chat format");

    std::cout << "Tokenizer tests passed\n";
    return 0;
}
