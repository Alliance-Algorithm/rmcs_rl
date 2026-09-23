#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <span>
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

inline bool model_id_of_file(const std::string& path, std::uint64_t& out, std::string& error) {
    std::ifstream stream{path, std::ios::binary};
    if (!stream) {
        error = "cannot open model file '" + path + "'";
        return false;
    }
    std::uint64_t hash = kFnv1a64Offset;
    std::string buffer(64 * 1024, '\0');
    while (stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto read = static_cast<std::size_t>(stream.gcount());
        if (read == 0)
            break;
        for (std::size_t i = 0; i < read; ++i)
            hash = (hash ^ static_cast<std::uint64_t>(static_cast<unsigned char>(buffer[i])))
                 * kFnv1a64Prime;
    }
    if (stream.bad()) {
        error = "failed while reading model file '" + path + "'";
        return false;
    }
    out = hash;
    return true;
}

} // namespace rmcs_rl
