#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <typeinfo>
#include <vector>

namespace rmcs_rl {

std::vector<std::string> split_by(const std::string& text, char delimiter);
std::vector<std::string> split_whitespace(const std::string& text);
std::string format_number(double value);
std::string join(const std::vector<std::string>& parts, const char* separator);
double parse_double(const std::string& text, const std::string& context);
std::int64_t parse_integer(const std::string& text, const std::string& context);
bool parse_boolean(const std::string& text, const std::string& context);
bool is_finite(double value);
std::string pretty_type(const std::type_info& type);

std::map<std::string, std::string> parse_tokens(const std::string& spec);
void validate_tokens(
    const std::map<std::string, std::string>& tokens, const std::vector<std::string>& allowed,
    const std::string& spec);
double double_token(
    const std::map<std::string, std::string>& tokens, const std::string& key, double fallback,
    const std::string& spec);
bool bool_token(
    const std::map<std::string, std::string>& tokens, const std::string& key, bool fallback,
    const std::string& spec);
std::string string_token(
    const std::map<std::string, std::string>& tokens, const std::string& key,
    const std::string& fallback);
void parse_clip(
    const std::map<std::string, std::string>& tokens, const std::string& spec, bool& has_clip,
    double& clip_min, double& clip_max);
void parse_index(
    const std::map<std::string, std::string>& tokens, const std::string& spec, bool& has_index,
    std::size_t& index);
std::vector<std::size_t>
parse_index_list(const std::string& text, std::size_t limit, const std::string& what);

} // namespace rmcs_rl
