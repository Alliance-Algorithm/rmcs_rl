#include <rmcs_rl/rl_bridge/utility.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

#include <eigen3/Eigen/Dense>
#include <rmcs_description/tf_description.hpp>

namespace rmcs_rl {

std::vector<std::string> split_by(const std::string& text, char delimiter) {
    std::vector<std::string> parts;
    std::string current;
    std::istringstream stream{text};
    while (std::getline(stream, current, delimiter))
        parts.push_back(current);
    return parts;
}

std::vector<std::string> split_whitespace(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream stream{text};
    std::string token;
    while (stream >> token)
        tokens.push_back(token);
    return tokens;
}

std::string format_number(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6g", value);
    return buffer;
}

std::string join(const std::vector<std::string>& parts, const char* separator) {
    std::string text;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i != 0)
            text += separator;
        text += parts[i];
    }
    return text;
}

double parse_double(const std::string& text, const std::string& context) {
    try {
        std::size_t consumed = 0;
        const double value = std::stod(text, &consumed);
        if (consumed != text.size())
            throw std::invalid_argument("trailing characters");
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument("RlBridge: '" + text + "' is not a number (" + context + ")");
    }
}

std::int64_t parse_integer(const std::string& text, const std::string& context) {
    try {
        std::size_t consumed = 0;
        const auto value = std::stoll(text, &consumed);
        if (consumed != text.size())
            throw std::invalid_argument("trailing characters");
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument("RlBridge: '" + text + "' is not an integer (" + context + ")");
    }
}

bool parse_boolean(const std::string& text, const std::string& context) {
    if (text == "true" || text == "1")
        return true;
    if (text == "false" || text == "0")
        return false;
    throw std::invalid_argument(
        "RlBridge: '" + text + "' is not a boolean (true/false) (" + context + ")");
}

bool is_finite(double value) { return std::isfinite(value); }

std::string pretty_type(const std::type_info& type) {
    if (type == typeid(double))
        return "double";
    if (type == typeid(float))
        return "float";
    if (type == typeid(bool))
        return "bool";
    if (type == typeid(int))
        return "int";
    if (type == typeid(std::size_t))
        return "std::size_t";
    if (type == typeid(Eigen::Vector3d))
        return "Eigen::Vector3d";
    if (type == typeid(rmcs_description::BaseLink::DirectionVector))
        return "rmcs_description::BaseLink::DirectionVector";
    if (type == typeid(Eigen::Quaterniond))
        return "Eigen::Quaterniond";
    return std::string{type.name()} + " (unknown)";
}

std::map<std::string, std::string> parse_tokens(const std::string& spec) {
    std::map<std::string, std::string> tokens;
    for (const auto& token : split_whitespace(spec)) {
        const auto equals = token.find('=');
        if (equals == std::string::npos || equals == 0)
            throw std::invalid_argument(
                "RlBridge: term token '" + token + "' must be key=value (term '" + spec + "')");
        const std::string key = token.substr(0, equals);
        const std::string value = token.substr(equals + 1);
        if (!tokens.emplace(key, value).second)
            throw std::invalid_argument(
                "RlBridge: duplicate key '" + key + "' in term '" + spec + "'");
    }
    return tokens;
}

void validate_tokens(
    const std::map<std::string, std::string>& tokens, const std::vector<std::string>& allowed,
    const std::string& spec) {
    for (const auto& [key, ignored] : tokens) {
        (void)ignored;
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
            throw std::invalid_argument(
                "RlBridge: unknown key '" + key + "' in term '" + spec + "'");
    }
}

double double_token(
    const std::map<std::string, std::string>& tokens, const std::string& key, double fallback,
    const std::string& spec) {
    const auto token = tokens.find(key);
    if (token == tokens.end())
        return fallback;
    const double value = parse_double(token->second, key + " in term '" + spec + "'");
    if (!is_finite(value))
        throw std::invalid_argument(
            "RlBridge: " + key + " must be finite (term '" + spec + "')");
    return value;
}

bool bool_token(
    const std::map<std::string, std::string>& tokens, const std::string& key, bool fallback,
    const std::string& spec) {
    const auto token = tokens.find(key);
    if (token == tokens.end())
        return fallback;
    return parse_boolean(token->second, key + " in term '" + spec + "'");
}

std::string string_token(
    const std::map<std::string, std::string>& tokens, const std::string& key,
    const std::string& fallback) {
    const auto token = tokens.find(key);
    if (token == tokens.end())
        return fallback;
    return token->second;
}

void parse_clip(
    const std::map<std::string, std::string>& tokens, const std::string& spec, bool& has_clip,
    double& clip_min, double& clip_max) {
    const auto clip = tokens.find("clip");
    if (clip == tokens.end())
        return;
    const auto separator = clip->second.find(':');
    if (separator == std::string::npos) {
        const double symmetric = parse_double(clip->second, "clip");
        if (!(symmetric > 0.0) || !is_finite(symmetric))
            throw std::invalid_argument(
                "RlBridge: single-value clip must be finite and > 0 (term '" + spec + "')");
        clip_min = -symmetric;
        clip_max = symmetric;
    } else {
        clip_min = parse_double(clip->second.substr(0, separator), "clip min");
        clip_max = parse_double(clip->second.substr(separator + 1), "clip max");
    }
    if (!is_finite(clip_min) || !is_finite(clip_max) || clip_min > clip_max)
        throw std::invalid_argument("RlBridge: invalid clip range (term '" + spec + "')");
    has_clip = true;
}

void parse_index(
    const std::map<std::string, std::string>& tokens, const std::string& spec, bool& has_index,
    std::size_t& index) {
    const auto token = tokens.find("index");
    if (token == tokens.end())
        return;
    const auto value = parse_integer(token->second, "index in term '" + spec + "'");
    if (value < 0)
        throw std::invalid_argument("RlBridge: index must be >= 0 (term '" + spec + "')");
    has_index = true;
    index = static_cast<std::size_t>(value);
}

std::vector<std::size_t>
parse_index_list(const std::string& text, std::size_t limit, const std::string& what) {
    std::vector<std::size_t> indices;
    for (const auto& piece : split_by(text, ',')) {
        if (piece.empty())
            throw std::invalid_argument("RlBridge: empty entry in " + what + " '" + text + "'");
        const auto range = piece.find("..");
        if (range == std::string::npos) {
            const auto index = parse_integer(piece, what);
            if (index < 0 || static_cast<std::size_t>(index) >= limit)
                throw std::invalid_argument(
                    "RlBridge: " + what + " index " + std::to_string(index)
                    + " out of range [0, " + std::to_string(limit) + ")");
            indices.push_back(static_cast<std::size_t>(index));
        } else {
            const auto first = parse_integer(piece.substr(0, range), what);
            const auto last = parse_integer(piece.substr(range + 2), what);
            if (first < 0 || last < first || static_cast<std::size_t>(last) >= limit)
                throw std::invalid_argument(
                    "RlBridge: invalid " + what + " range '" + piece + "' (limit "
                    + std::to_string(limit) + ")");
            for (auto index = first; index <= last; ++index)
                indices.push_back(static_cast<std::size_t>(index));
        }
    }
    return indices;
}

} // namespace rmcs_rl
