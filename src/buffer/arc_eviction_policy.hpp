#pragma once
#include "buffer/eviction_policy.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace shilmandb {

// The sizes of ARC's four lists and its target p, for tests
struct ArcSizes {
    size_t t1, t2, b1, b2;
    double p;
};

// ARC, the Adaptive Replacement Cache
class ArcEvictionPolicy : public EvictionPolicy {
public:
    explicit ArcEvictionPolicy(size_t num_frames);

    // Throws: the buffer pool always passes the page
    void RecordAccess(frame_id_t frame_id) override;

    // A new use
    void RecordAccess(frame_id_t frame_id, page_id_t page_id) override;
    void RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType type) override;
    void SetEvictable(frame_id_t frame_id, bool evictable) override;

    // A victim for a page never seen
    [[nodiscard]] std::optional<frame_id_t> Evict() override;
    [[nodiscard]] std::optional<frame_id_t> Evict(page_id_t incoming_page) override;
    void Remove(frame_id_t frame_id) override;

    [[nodiscard]] size_t Size() const override;
    [[nodiscard]] bool UsesAccessTypes() const override { return true; }
    void Reset() override;

    [[nodiscard]] ArcSizes Sizes() const;

private:
    enum List : uint8_t { kT1, kT2, kB1, kB2, kOut };

    static constexpr page_id_t kNone = INVALID_PAGE_ID;
    static constexpr frame_id_t kNoFrame = std::numeric_limits<frame_id_t>::max();

    // A page's place in the lists, threaded through page ids. The front of a list is its oldest end.
    struct PageState {
        page_id_t prev{kNone};
        page_id_t next{kNone};
        frame_id_t frame{kNoFrame};  // its frame while in T1 or T2
        List list{kOut};
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

    [[nodiscard]] List ListOf(page_id_t page_id) const;
    [[nodiscard]] size_t Count(List list) const { return ends_[list].size; }

    void PushBack(page_id_t page_id, List list);
    void Unlink(page_id_t page_id);
    void MoveTo(page_id_t page_id, List list);

    // The paper's p updates for a hit in B1 and in B2
    void GrowP();
    void ShrinkP();

    [[nodiscard]] page_id_t OldestEvictable(List list) const;

    // The oldest evictable page of `list`, else of the other cache list; throws if neither has one
    [[nodiscard]] page_id_t PickVictim(List list) const;

    // Forgets the oldest ghost of B1 or B2; throws if the list is empty
    void DropOldestGhost(List list);

    // Takes the page out of the cache: its frame stops being tracked
    frame_id_t Untrack(page_id_t page_id);

    // REPLACE: evicts the oldest page of T1 or T2 into its ghost list and returns its frame
    frame_id_t Replace(bool incoming_in_b2);
    
    // What a load nobody prepared does before the insert: move p or make room
    void MakeRoom(List incoming_list);

    size_t c_;
    std::vector<FrameState> frames_;  // one entry per frame, indexed by frame id
    std::vector<PageState> pages_;    // one entry per page id, grown on demand
    std::array<Ends, 4> ends_{};      // T1, T2, B1, B2
    double p_{0};
    size_t evictable_count_{0};

    // The last Evict, waiting for its load
    frame_id_t prepared_frame_{kNoFrame};
    page_id_t prepared_page_{INVALID_PAGE_ID};
};

}  // namespace shilmandb
