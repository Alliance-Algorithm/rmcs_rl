#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

namespace rmcs_rl {

inline constexpr std::uint64_t kFnv1a64Offset = 0xCBF29CE484222325ULL;
inline constexpr std::uint64_t kFnv1a64Prime = 0x100000001B3ULL;

inline std::uint64_t fnv1a64(std::string_view bytes) {
    std::uint64_t hash = kFnv1a64Offset;
    for (const char raw : bytes) {
        hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(raw));
        hash *= kFnv1a64Prime;
    }
    return hash;
}

inline std::string hex16(std::uint64_t value) {
    char buffer[17];
    std::snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(value));
    return buffer;
}

inline std::uint64_t layout_hash(
    std::string_view obs_signature, std::string_view actions_signature, std::size_t obs_size,
    std::size_t actions_size) {
    std::string canonical;
    canonical.reserve(obs_signature.size() + actions_signature.size() + 48);
    canonical.append(obs_signature);
    canonical.append("||");
    canonical.append(actions_signature);
    canonical.append("||");
    canonical.append(std::to_string(obs_size));
    canonical.append("x");
    canonical.append(std::to_string(actions_size));
    return fnv1a64(canonical);
}

} // namespace rmcs_rl
