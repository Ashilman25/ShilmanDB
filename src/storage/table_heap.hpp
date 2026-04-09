#pragma once
#include "buffer/buffer_pool_manager.hpp"
#include "common/rid.hpp"
#include "types/tuple.hpp"
#include "catalog/schema.hpp"
#include "common/config.hpp"

namespace shilmandb {

class TableHeap {
public:
    // Open an existing heap file by its first page id
    TableHeap(BufferPoolManager* bpm, page_id_t first_page_id);

    // Create a new empty heap
    explicit TableHeap(BufferPoolManager* bpm);

    [[nodiscard]] RID InsertTuple(const Tuple& tuple);
    [[nodiscard]] bool GetTuple(const RID& rid, Tuple* tuple, const Schema& schema);
    [[nodiscard]] bool DeleteTuple(const RID& rid);

    [[nodiscard]] page_id_t GetFirstPageId() const;

    class Iterator {
    public:
        Iterator(TableHeap* table, RID rid, const Schema* schema);

        Tuple operator*();
        Iterator& operator++();
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;
        [[nodiscard]] RID GetRID() const;

    private:
        TableHeap* table_;
        RID current_rid_;
        const Schema* schema_;

        void AdvancePastDeleted();
    };

    Iterator Begin(const Schema& schema);
    Iterator End();

private:
    BufferPoolManager* bpm_;
    page_id_t first_page_id_;
    page_id_t last_insert_page_id_;
};

}  // namespace shilmandb
