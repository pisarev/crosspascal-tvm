// Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
#include "sentencepiece_processor.h"
#include "builtin_pb/sentencepiece_model.pb.h"

#include <algorithm>
#include <climits>
#include <memory>
#include <string>
#include <vector>

#define API extern "C" __declspec(dllexport)
using sentencepiece::SentencePieceProcessor;

API int cp_sp_build_bpe(const char* const* pieces, const float* scores, const int* kinds,
                        int count, const int* special_ids, int add_prefix,
                        char* output, int capacity) noexcept {
    try {
        if (!pieces || !scores || !kinds || !special_ids || count <= 0 || capacity < 0)
            return -1;
        for (int i = 0; i < 4; ++i) {
            if (special_ids[i] < -1 || special_ids[i] >= count)
                return -1;
            if (special_ids[i] >= 0 && !pieces[special_ids[i]])
                return -1;
        }
        if (special_ids[0] < 0)
            return -1;

        sentencepiece::ModelProto model;
        auto* trainer = model.mutable_trainer_spec();
        trainer->set_model_type(sentencepiece::TrainerSpec::BPE);
        trainer->set_byte_fallback(true);
        trainer->set_unk_id(special_ids[0]);
        trainer->set_bos_id(special_ids[1]);
        trainer->set_eos_id(special_ids[2]);
        trainer->set_pad_id(special_ids[3]);
        trainer->set_unk_piece(pieces[special_ids[0]]);
        if (special_ids[1] >= 0) trainer->set_bos_piece(pieces[special_ids[1]]);
        if (special_ids[2] >= 0) trainer->set_eos_piece(pieces[special_ids[2]]);
        if (special_ids[3] >= 0) trainer->set_pad_piece(pieces[special_ids[3]]);

        auto* normalizer = model.mutable_normalizer_spec();
        normalizer->set_name("identity");
        normalizer->set_add_dummy_prefix(add_prefix != 0);
        normalizer->set_remove_extra_whitespaces(false);

        for (int i = 0; i < count; ++i) {
            using Piece = sentencepiece::ModelProto::SentencePiece;
            if (!pieces[i] || !Piece::Type_IsValid(kinds[i])) return -1;
            auto* entry = model.add_pieces();
            entry->set_piece(pieces[i]);
            entry->set_score(scores[i]);
            entry->set_type(static_cast<Piece::Type>(kinds[i]));
        }
        const size_t size = model.ByteSizeLong();
        if (size > INT_MAX) return -1;
        if (output && size <= static_cast<size_t>(capacity) &&
            !model.SerializeToArray(output, static_cast<int>(size)))
            return -1;
        return static_cast<int>(size);
    } catch (...) {
        return -1;
    }
}

API SentencePieceProcessor* cp_sp_load(const char* bytes, int size) noexcept {
    try {
        if (!bytes || size <= 0) return nullptr;
        auto processor = std::make_unique<SentencePieceProcessor>();
        if (!processor->LoadFromSerializedProto({bytes, static_cast<size_t>(size)}).ok())
            return nullptr;
        return processor.release();
    } catch (...) {
        return nullptr;
    }
}

API void cp_sp_free(SentencePieceProcessor* processor) noexcept {
    delete processor;
}

API int cp_sp_id(const SentencePieceProcessor* processor, const char* piece) noexcept {
    try {
        const int id = processor->PieceToId(piece);
        return id >= 0 && processor->IdToPiece(id) == piece ? id : -1;
    } catch (...) {
        return -1;
    }
}

API int cp_sp_encode(const SentencePieceProcessor* processor, const char* text,
                     int size, int* output, int capacity) noexcept {
    try {
        std::vector<int> ids;
        if (!processor->Encode({text, static_cast<size_t>(size)}, &ids).ok() ||
            ids.size() > INT_MAX)
            return -1;
        if (output && ids.size() <= static_cast<size_t>(capacity))
            std::copy(ids.begin(), ids.end(), output);
        return static_cast<int>(ids.size());
    } catch (...) {
        return -1;
    }
}

API int cp_sp_decode(const SentencePieceProcessor* processor, const int* ids,
                     int count, char* output, int capacity) noexcept {
    try {
        if (count == 0) return 0;
        std::string text;
        if (!processor->Decode(std::vector<int>(ids, ids + count), &text).ok() ||
            text.size() > INT_MAX)
            return -1;
        if (output && text.size() <= static_cast<size_t>(capacity))
            std::copy(text.begin(), text.end(), output);
        return static_cast<int>(text.size());
    } catch (...) {
        return -1;
    }
}
