#pragma once
#include "catalog/schema.hpp"
#include "parser/ast.hpp"
#include <utility>
#include <vector>
#include <memory>
#include <string>
#include <optional>

namespace shilmandb {


enum class PlanNodeType {
    SEQ_SCAN, INDEX_SCAN, FILTER, HASH_JOIN, NESTED_LOOP_JOIN,
    SORT, AGGREGATE, PROJECTION, LIMIT
};

struct PlanNode {
    PlanNodeType type;
    Schema output_schema;
    std::vector<std::unique_ptr<PlanNode>> children;

    virtual ~PlanNode() = default;

protected:
    PlanNode(PlanNodeType t, Schema schema) : type(t), output_schema(std::move(schema)) {}
};

struct SeqScanPlanNode : PlanNode {
    std::string table_name;
    SeqScanPlanNode(Schema schema, std::string tbl) : PlanNode(PlanNodeType::SEQ_SCAN, std::move(schema)), table_name(std::move(tbl)) {}
};

struct IndexScanPlanNode : PlanNode {
    std::string table_name;
    std::string index_name;
    std::optional<Value> low_key;
    std::optional<Value> high_key;

    IndexScanPlanNode(Schema schema, std::string tbl, std::string idx) : PlanNode(PlanNodeType::INDEX_SCAN, std::move(schema)), table_name(std::move(tbl)), index_name(std::move(idx)) {}
};

struct FilterPlanNode : PlanNode {
    std::unique_ptr<Expression> predicate;
    FilterPlanNode(Schema schema, std::unique_ptr<Expression> pred) : PlanNode(PlanNodeType::FILTER, std::move(schema)), predicate(std::move(pred)) {}
};

struct HashJoinPlanNode : PlanNode {
    std::unique_ptr<Expression> join_predicate;
    // Left child = build side, Right child = probe side
    HashJoinPlanNode(Schema schema, std::unique_ptr<Expression> pred) : PlanNode(PlanNodeType::HASH_JOIN, std::move(schema)), join_predicate(std::move(pred)) {}
};

struct NestedLoopJoinPlanNode : PlanNode {
    std::unique_ptr<Expression> predicate;
    NestedLoopJoinPlanNode(Schema schema, std::unique_ptr<Expression> pred) : PlanNode(PlanNodeType::NESTED_LOOP_JOIN, std::move(schema)), predicate(std::move(pred)) {}
};

struct SortPlanNode : PlanNode {
    std::vector<OrderByItem> order_by;
    SortPlanNode(Schema schema, std::vector<OrderByItem> ob) : PlanNode(PlanNodeType::SORT, std::move(schema)), order_by(std::move(ob)) {}
};

struct AggregatePlanNode : PlanNode {
    std::vector<std::unique_ptr<Expression>> group_by_exprs;
    std::vector<std::unique_ptr<Expression>> aggregate_exprs;
    std::vector<Aggregate::Func> aggregate_funcs;

    AggregatePlanNode(Schema schema) : PlanNode(PlanNodeType::AGGREGATE, std::move(schema)) {}
};

struct ProjectionPlanNode : PlanNode {
    std::vector<std::unique_ptr<Expression>> expressions;
    ProjectionPlanNode(Schema schema, std::vector<std::unique_ptr<Expression>> exprs) : PlanNode(PlanNodeType::PROJECTION, std::move(schema)), expressions(std::move(exprs)) {}
};

struct LimitPlanNode : PlanNode {
    int64_t limit;
    LimitPlanNode(Schema schema, int64_t lim) : PlanNode(PlanNodeType::LIMIT, std::move(schema)), limit(lim) {}
};

} //namespace shilmandb