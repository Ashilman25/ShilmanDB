#pragma once
#include "buffer/eviction_policy.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace shilmandb {

// The sizes of LIRS's sets, for tests
struct LirsSizes {
    size_t lir;    // LIR pages, always resident
    size_t hir;    // resident HIR pages: the queue Q
    size_t stack;  // entries in the stack S, resident or not
};

// LIRS, the Low Inter-reference Recency Set policy
class LirsEvictionPolicy : public EvictionPolicy {
public:
    // Throws for a pool under 2 frames: LIRS needs a frame for LIR pages and one for HIR
    explicit LirsEvictionPolicy(size_t num_frames);

    // The incoming page doesn't change the victim
    using EvictionPolicy::Evict;

    // Throws: the buffer pool always passes the page
    void RecordAccess(frame_id_t frame_id) override;
    // A new use
    void RecordAccess(frame_id_t frame_id, page_id_t page_id) override;
    void RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType type) override;
    void SetEvictable(frame_id_t frame_id, bool evictable) override;

    [[nodiscard]] std::optional<frame_id_t> Evict() override;
    void Remove(frame_id_t frame_id) override;

    [[nodiscard]] size_t Size() const override;
    [[nodiscard]] bool UsesAccessTypes() const override { return true; }
    void Reset() override;

    [[nodiscard]] LirsSizes Sizes() const;

private:
    enum ListId : uint8_t { kS, kQ };

    static constexpr page_id_t kNone = INVALID_PAGE_ID;
    static constexpr frame_id_t kNoFrame = std::numeric_limits<frame_id_t>::max();

    struct Links {
        page_id_t prev{kNone};
        page_id_t next{kNone};
    };
    // A page's place in S and Q, threaded through page ids. S's front is its bottom; Q's
    // front is the next victim.
    struct PageState {
        std::array<Links, 2> links{};  // in S, in Q
        frame_id_t frame{kNoFrame};    // its frame while resident
        bool lir{false};
        bool in_s{false};
    };
    struct Ends {
        page_id_t head{kNone};
        page_id_t tail{kNone};
        size_t size{0};
    };
    struct FrameState {
        page_id_t page{INVALID_PAGE_ID};  // INVALID_PAGE_ID while the frame isn't tracked
        bool evictable{false};
    };

    // Throws if the frame id is past the last frame
    void CheckRange(frame_id_t frame_id) const;

    void PushBack(page_id_t page_id, ListId list);
    void Unlink(page_id_t page_id, ListId list);

    void MoveToTop(page_id_t page_id);
    // Drops HIR entries from the bottom of S until an LIR page is there
    void Prune();
    // The LIR page at the bottom of S becomes a resident HIR page at the back of Q
    void DemoteBottom();

    // What a touch, a new use of a resident page, and a load do
    void Touch(page_id_t page_id);
    void Hit(page_id_t page_id);
    void Load(page_id_t page_id);

    // Makes a resident page non-resident, the way an eviction does: a HIR page leaves Q and
    // keeps its place in S; an LIR page leaves S and the LIR set
    void Unload(page_id_t page_id);
    // Evicts a resident page and returns its frame, which stops being tracked
    frame_id_t EvictPage(page_id_t page_id);

    size_t lir_limit_;                // most LIR pages the pool holds
    std::vector<FrameState> frames_;  // one entry per frame, indexed by frame id
    std::vector<PageState> pages_;    // one entry per page id, grown on demand
    std::array<Ends, 2> ends_{};      // S, Q
    size_t lir_count_{0};
    size_t evictable_count_{0};
};

}  // namespace shilmandb
