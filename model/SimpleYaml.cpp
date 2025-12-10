#include "SimpleYaml.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
struct StackEntry {
    int indent;
    SimpleYaml::Node* node;
};
} // namespace

const SimpleYaml::Node& SimpleYaml::Node::at(const std::string& key) const {
    auto it = children.find(key);
    if (it == children.end()) {
        throw std::runtime_error("Missing YAML key: " + key);
    }
    return it->second;
}

SimpleYaml::Node& SimpleYaml::Node::ensureMapChild(const std::string& key) {
    type = Type::Map;
    return children[key];
}

double SimpleYaml::Node::asScalar() const {
    if (type != Type::Scalar) {
        throw std::runtime_error("Requested scalar from non-scalar YAML node");
    }
    if (!hasNumeric) {
        throw std::runtime_error("Requested numeric scalar from non-numeric YAML node");
    }
    return scalar;
}

const std::string& SimpleYaml::Node::asString() const {
    if (type != Type::Scalar) {
        throw std::runtime_error("Requested string from non-scalar YAML node");
    }
    return text;
}

const std::vector<double>& SimpleYaml::Node::asSequence() const {
    if (type != Type::Sequence) {
        throw std::runtime_error("Requested list from non-sequence YAML node");
    }
    return sequence;
}

bool SimpleYaml::Node::asBool() const {
    if (type != Type::Scalar) {
        throw std::runtime_error("Requested bool from non-scalar YAML node");
    }
    // Check for common boolean representations
    std::string lower = text;
    for (char& c : lower) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (lower == "true" || lower == "yes" || lower == "on" || lower == "1") {
        return true;
    }
    if (lower == "false" || lower == "no" || lower == "off" || lower == "0") {
        return false;
    }
    throw std::runtime_error("Cannot parse as boolean: " + text);
}

SimpleYaml::SimpleYaml(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open YAML file: " + path);
    }

    std::string line;
    std::vector<StackEntry> stack;
    stack.push_back({-2, &root_});

    while (std::getline(file, line)) {
        const auto hashPos = line.find('#');
        if (hashPos != std::string::npos) {
            line = line.substr(0, hashPos);
        }

        auto trimmed = trim(line);
        if (trimmed.empty()) {
            continue;
        }

        const int indent = indentation(line);
        while (!stack.empty() && indent <= stack.back().indent) {
            stack.pop_back();
        }
        if (stack.empty()) {
            throw std::runtime_error("Invalid indentation in YAML file");
        }

        const auto colon = trimmed.find(':');
        if (colon == std::string::npos) {
            throw std::runtime_error("Invalid line (missing colon): " + trimmed);
        }

        const auto key = trim(trimmed.substr(0, colon));
        std::string value = trim(trimmed.substr(colon + 1));

        SimpleYaml::Node* parent = stack.back().node;
        SimpleYaml::Node& node = parent->ensureMapChild(key);

        if (value.empty()) {
            node.type = Node::Type::Map;
            stack.push_back({indent, &node});
            continue;
        }

        if (!value.empty() && value.front() == '[') {
            node.type = Node::Type::Sequence;
            node.sequence = parseList(value);
        } else {
            node.type = Node::Type::Scalar;
            node.text = value;
            try {
                node.scalar = std::stod(value);
                node.hasNumeric = true;
            } catch (const std::invalid_argument&) {
                node.scalar = 0.0;
                node.hasNumeric = false;
            }
        }
    }
}

const SimpleYaml::Node& SimpleYaml::nodeAtPath(const std::string& dottedPath) const {
    const Node* node = &root_;
    std::stringstream ss(dottedPath);
    std::string item;
    while (std::getline(ss, item, '.')) {
        if (item.empty()) {
            continue;
        }
        node = &node->at(item);
    }
    return *node;
}

std::string SimpleYaml::trim(const std::string& str) {
    size_t start = 0;
    while (start < str.size() && std::isspace(static_cast<unsigned char>(str[start]))) {
        ++start;
    }
    if (start == str.size()) {
        return {};
    }
    size_t end = str.size() - 1;
    while (end > start && std::isspace(static_cast<unsigned char>(str[end]))) {
        --end;
    }
    return str.substr(start, end - start + 1);
}

int SimpleYaml::indentation(const std::string& line) {
    int count = 0;
    for (char c : line) {
        if (c == ' ') {
            ++count;
        } else {
            break;
        }
    }
    return count;
}

std::vector<double> SimpleYaml::parseList(const std::string& text) {
    std::vector<double> values;
    auto stripped = text;
    if (stripped.front() == '[') {
        stripped = stripped.substr(1);
    }
    if (!stripped.empty() && stripped.back() == ']') {
        stripped.pop_back();
    }

    std::stringstream ss(stripped);
    std::string number;
    while (std::getline(ss, number, ',')) {
        auto trimmed = trim(number);
        if (!trimmed.empty()) {
            values.push_back(std::stod(trimmed));
        }
    }
    return values;
}
