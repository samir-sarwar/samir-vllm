// AI generated test

#include "tokenizer.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace
{

    void expect_ids(const std::vector<int> &actual, const std::vector<int> &expected, const std::string &name)
    {
        if (actual == expected)
        {
            return;
        }
        std::cerr << name << " failed\n";
        std::cerr << "Expected: ";
        for (int id : expected)
            std::cerr << id << ' ';
        std::cerr << "\nActual:   ";
        for (int id : actual)
            std::cerr << id << ' ';
        std::cerr << '\n';
        std::exit(1);
    }

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: tokenizer-test path/to/tokenizer.model\n";
        return 1;
    }

    Tokenizer tokenizer;
    std::string error;
    if (!tokenizer.load(argv[1], &error))
    {
        std::cerr << error << '\n';
        return 1;
    }

    if (tokenizer.base_vocab_size() != 128000)
    {
        std::cerr << "Wrong vocabulary size\n";
        return 1;
    }

    expect_ids(
        tokenizer.encode("This is a test sentence.", true, true),
        {128000, 2028, 374, 264, 1296, 11914, 13, 128001},
        "Meta basic encode example");
    expect_ids(tokenizer.encode("Hello, world!", false), {9906, 11, 1917, 0}, "basic punctuation");
    expect_ids(
        tokenizer.encode("We're testing 123456 and don't stop.", false),
        {1687, 2351, 7649, 220, 4513, 10961, 323, 1541, 956, 3009, 13},
        "contractions and numbers");
    expect_ids(
        tokenizer.encode("  spaces\tand\nnewlines  ", false),
        {220, 12908, 53577, 198, 943, 8128, 256},
        "whitespace");

    std::cout << "Tokenizer tests passed\n";
    return 0;
}
