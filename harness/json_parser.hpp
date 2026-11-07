// Lightweight, zero-dependency JSON parser for harness manifests and configs.
#pragma once

#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <iomanip>

#include "harness/harness_types.hpp"

namespace stitch::harness {

class JsonValue {
public:
    enum Type { Null, Bool, Number, String, Array, Object };

    Type type{Null};
    bool bool_val{false};
    double num_val{0.0};
    std::string str_val;
    std::vector<JsonValue> arr_val;
    std::map<std::string, JsonValue> obj_val;

    static JsonValue parse(const std::string& text) {
        std::size_t idx = 0;
        skip_ws(text, idx);
        JsonValue value = parse_value(text, idx);
        skip_ws(text, idx);
        if (idx != text.size()) throw std::runtime_error("Trailing content after JSON value");
        return value;
    }

    static JsonValue parse_file(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open JSON file: " + path);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        return parse(ss.str());
    }

private:
    static void skip_ws(const std::string& s, std::size_t& i) {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
            ++i;
        }
    }

    static JsonValue parse_value(const std::string& s, std::size_t& i) {
        skip_ws(s, i);
        if (i >= s.size()) throw std::runtime_error("Unexpected end of JSON");

        char c = s[i];
        if (c == '{') return parse_object(s, i);
        if (c == '[') return parse_array(s, i);
        if (c == '"') return parse_string(s, i);
        if (c == 't' || c == 'f') return parse_bool(s, i);
        if (c == 'n') return parse_null(s, i);
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number(s, i);

        throw std::runtime_error(std::string("Unexpected character in JSON: ") + c);
    }

    static JsonValue parse_object(const std::string& s, std::size_t& i) {
        JsonValue val;
        val.type = Object;
        ++i; // consume '{'
        skip_ws(s, i);

        if (i < s.size() && s[i] == '}') {
            ++i;
            return val;
        }

        while (i < s.size()) {
            skip_ws(s, i);
            if (s[i] != '"') throw std::runtime_error("Expected string key in object");
            JsonValue key = parse_string(s, i);
            skip_ws(s, i);
            if (i >= s.size() || s[i] != ':') throw std::runtime_error("Expected ':' after key");
            ++i; // consume ':'
            JsonValue child = parse_value(s, i);
            if (!val.obj_val.emplace(key.str_val, std::move(child)).second)
                throw std::runtime_error("Duplicate JSON key: " + key.str_val);

            skip_ws(s, i);
            if (i < s.size() && s[i] == ',') {
                ++i;
                skip_ws(s, i);
            } else if (i < s.size() && s[i] == '}') {
                ++i;
                return val;
            } else {
                throw std::runtime_error("Expected ',' or '}' in object");
            }
        }
        throw std::runtime_error("Unterminated object");
    }

    static JsonValue parse_array(const std::string& s, std::size_t& i) {
        JsonValue val;
        val.type = Array;
        ++i; // consume '['
        skip_ws(s, i);

        if (i < s.size() && s[i] == ']') {
            ++i;
            return val;
        }

        while (i < s.size()) {
            val.arr_val.push_back(parse_value(s, i));
            skip_ws(s, i);
            if (i < s.size() && s[i] == ',') {
                ++i;
                skip_ws(s, i);
            } else if (i < s.size() && s[i] == ']') {
                ++i;
                return val;
            } else {
                throw std::runtime_error("Expected ',' or ']' in array");
            }
        }
        throw std::runtime_error("Unterminated array");
    }

    static JsonValue parse_string(const std::string& s, std::size_t& i) {
        JsonValue val;
        val.type = String;
        ++i; // consume '"'
        std::string res;
        while (i < s.size()) {
            char c = s[i++];
            if (c == '"') {
                val.str_val = res;
                return val;
            }
            if (c == '\\' && i < s.size()) {
                char esc = s[i++];
                if (esc == '"' || esc == '\\' || esc == '/') res += esc;
                else if (esc == 'b') res += '\b';
                else if (esc == 'f') res += '\f';
                else if (esc == 'n') res += '\n';
                else if (esc == 'r') res += '\r';
                else if (esc == 't') res += '\t';
                else res += esc;
            } else {
                res += c;
            }
        }
        throw std::runtime_error("Unterminated string");
    }

    static JsonValue parse_number(const std::string& s, std::size_t& i) {
        JsonValue val;
        val.type = Number;
        std::size_t start = i;
        if (s[i] == '-') ++i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        if (i < s.size() && s[i] == '.') {
            ++i;
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            ++i;
            if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        }
        val.num_val = std::stod(s.substr(start, i - start));
        return val;
    }

    static JsonValue parse_bool(const std::string& s, std::size_t& i) {
        JsonValue val;
        val.type = Bool;
        if (s.compare(i, 4, "true") == 0) {
            val.bool_val = true;
            i += 4;
            return val;
        }
        if (s.compare(i, 5, "false") == 0) {
            val.bool_val = false;
            i += 5;
            return val;
        }
        throw std::runtime_error("Invalid boolean value");
    }

    static JsonValue parse_null(const std::string& s, std::size_t& i) {
        if (s.compare(i, 4, "null") == 0) {
            i += 4;
            JsonValue val;
            val.type = Null;
            return val;
        }
        throw std::runtime_error("Invalid null value");
    }
};

inline std::vector<PairEntry> parse_manifest(const std::string& path) {
    JsonValue root = JsonValue::parse_file(path);
    if (root.type != JsonValue::Object) {
        throw std::runtime_error("Manifest root must be a JSON object");
    }
    std::vector<PairEntry> pairs;
    for (const auto& kv : root.obj_val) {
        const auto& pair_id = kv.first;
        const auto& entry = kv.second;
        if (entry.type != JsonValue::Object) continue;
        PairEntry pe;
        pe.pair_id = pair_id;
        auto it_a = entry.obj_val.find("a");
        auto it_b = entry.obj_val.find("b");
        if (it_a == entry.obj_val.end() || it_b == entry.obj_val.end()) {
            throw std::runtime_error("Manifest entry missing 'a' or 'b' for pair: " + pair_id);
        }
        pe.img_a_path = it_a->second.str_val;
        pe.img_b_path = it_b->second.str_val;
        auto it_ref = entry.obj_val.find("reference_status");
        if (it_ref != entry.obj_val.end()) {
            pe.reference_status = it_ref->second.str_val;
        }
        pairs.push_back(pe);
    }
    return pairs;
}

inline std::vector<HarnessConfig> parse_configs(const std::string& path) {
    JsonValue root = JsonValue::parse_file(path);
    if (root.type != JsonValue::Array) {
        throw std::runtime_error("Configs root must be a JSON array");
    }
    std::vector<HarnessConfig> configs;
    for (const auto& item : root.arr_val) {
        if (item.type != JsonValue::Object) continue;
        HarnessConfig cfg;
        auto it_id = item.obj_val.find("config_id");
        auto it_s = item.obj_val.find("S");
        auto it_k = item.obj_val.find("K");
        auto it_r = item.obj_val.find("R");
        auto it_b = item.obj_val.find("blend_mode");

        if (it_s != item.obj_val.end()) cfg.S = it_s->second.num_val;
        if (it_k != item.obj_val.end()) cfg.K = static_cast<int>(it_k->second.num_val);
        if (it_r != item.obj_val.end()) cfg.R = static_cast<int>(it_r->second.num_val);
        if (it_b != item.obj_val.end()) {
            cfg.blend_mode_str = it_b->second.str_val;
            if (cfg.blend_mode_str == "multiband3") cfg.blend_mode = s6::BlendMode::Multiband3;
            else if (cfg.blend_mode_str == "multiband5") cfg.blend_mode = s6::BlendMode::Multiband5;
            else cfg.blend_mode = s6::BlendMode::Feather;
        }

        if (it_id != item.obj_val.end() && !it_id->second.str_val.empty()) {
            cfg.config_id = it_id->second.str_val;
        } else {
            std::stringstream ss;
            ss << "S" << std::fixed << std::setprecision(2) << cfg.S
               << "_K" << cfg.K
               << "_R" << cfg.R
               << "_B" << cfg.blend_mode_str;
            cfg.config_id = ss.str();
        }

        configs.push_back(cfg);
    }
    return configs;
}

}  // namespace stitch::harness
