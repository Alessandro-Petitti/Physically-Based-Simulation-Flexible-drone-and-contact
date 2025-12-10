#pragma once

#include <map>
#include <string>
#include <vector>

/**
 * A super lightweight YAML reader that only understands the subset of YAML used
 * by the drone parameter file. It supports nested maps with indentation-based
 * scopes, scalar floating point values and inline numeric arrays written as
 * [a, b, c]. That's sufficient for the constant data in the Python model and
 * keeps us self-contained without external dependencies.
 */
class SimpleYaml {
public:
    struct Node {
        enum class Type { Map, Scalar, Sequence };

        Type type{Type::Map};
        double scalar{};
        bool hasNumeric{false};
        std::string text;
        std::vector<double> sequence;
        std::map<std::string, Node> children;

        const Node& at(const std::string& key) const;
        Node& ensureMapChild(const std::string& key);
        double asScalar() const;
        const std::string& asString() const;
        const std::vector<double>& asSequence() const;
        bool asBool() const;
    };

    explicit SimpleYaml(const std::string& path);

    const Node& root() const { return root_; }
    const Node& nodeAtPath(const std::string& dottedPath) const;

private:
    Node root_;

    static std::string trim(const std::string& str);
    static int indentation(const std::string& line);
    static std::vector<double> parseList(const std::string& text);
};
