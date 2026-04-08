#include "types/value.hpp"
#include "common/exception.hpp"
#include <cstring>
#include <sstream>
#include <iomanip>
#include <limits>

namespace {

//date helpers
constexpr int kBaseYear = 1970;

bool IsLeapYear(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

int DaysInMonth(int month, int year) {
    constexpr int kDays[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2 && IsLeapYear(year)) return 29;
    return kDays[month];
}


int DaysInYear(int year) {
    return IsLeapYear(year) ? 366 : 365;
}

//convert days since base year to (year, month, day)
void DaysToDate(int32_t days, int& year, int& month, int& day) {
    // Handle negative days (before epoch)
    year = kBaseYear;

    if (days >= 0) {
        while (days >= DaysInYear(year)) {
            days -= DaysInYear(year);
            ++year;
        }
    } else {
        while (days < 0) {
            --year;
            days += DaysInYear(year);
        }
    }


    month = 1;
    while (days >= DaysInMonth(month, year)) {
        days -= DaysInMonth(month, year);
        ++month;
    }

    day = days + 1;  
}

//convert (year, month, day) to days since base year
int32_t DateToDays(int year, int month, int day) {
    int32_t total = 0;

    if (year >= kBaseYear) {
        for (int y = kBaseYear; y < year; ++y) {
            total += DaysInYear(y);
        }
    } else {
        for (int y = year; y < kBaseYear; ++y) {
            total -= DaysInYear(y);
        }
    }

    for (int m = 1; m < month; ++m) {
        total += DaysInMonth(m, year);
    }
    total += (day - 1);
    return total;
}


} //date helpers namespace



namespace shilmandb {

//constructors

Value::Value() : type_{TypeId::INVALID}, bigint_{0} {}

Value::Value(int32_t val) : type_{TypeId::INTEGER}, integer_{val} {}

Value::Value(int64_t val) : type_{TypeId::BIGINT}, bigint_{val} {}

Value::Value(double val) : type_{TypeId::DECIMAL}, decimal_{val} {}

Value::Value(const std::string& val) : type_{TypeId::VARCHAR}, bigint_{0}, varchar_{val} {}

Value Value::MakeDate(int32_t days_since_epoch) {
    Value v;
    v.type_ = TypeId::DATE;
    v.date_ = days_since_epoch;
    return v;
}

//compare operators

bool Value::operator==(const Value& other) const {
    if (type_ != other.type_) {
        throw DatabaseException("Type mismatch in comparison");
    }
    switch (type_) {
        case TypeId::INTEGER: 
            return integer_ == other.integer_;
        case TypeId::BIGINT:  
            return bigint_ == other.bigint_;
        case TypeId::DECIMAL: 
            return decimal_ == other.decimal_;
        case TypeId::VARCHAR:
            return varchar_ == other.varchar_;
        case TypeId::DATE:
            return date_ == other.date_;
        case TypeId::INVALID: 
            return true;  // NULL == NULL for hashmap key
    }

    return false;
}

bool Value::operator!=(const Value& other) const { 
    return !(*this == other); 
}

bool Value::operator<(const Value& other) const {
    if (type_ != other.type_) {
        throw DatabaseException("Type mismatch in comparison");
    }

    switch (type_) {
        case TypeId::INTEGER: 
            return integer_ < other.integer_;
        case TypeId::BIGINT:  
            return bigint_ < other.bigint_;
        case TypeId::DECIMAL: 
            return decimal_ < other.decimal_;
        case TypeId::VARCHAR: 
            return varchar_ < other.varchar_;
        case TypeId::DATE:    
            return date_ < other.date_;
        case TypeId::INVALID: 
            return false;
    }

    return false;
}

bool Value::operator>(const Value& other) const  { return other < *this; }
bool Value::operator<=(const Value& other) const { return !(other < *this); }
bool Value::operator>=(const Value& other) const { return !(*this < other); }



Value Value::Add(const Value& other) const {
    if (type_ != other.type_) {
        throw DatabaseException("Type mismatch in arithmetic");
    }
    switch (type_) {
        case TypeId::INTEGER:
            return Value(integer_ + other.integer_);
        case TypeId::BIGINT:  
            return Value(bigint_ + other.bigint_);
        case TypeId::DECIMAL: 
            return Value(decimal_ + other.decimal_);
        default:
            throw DatabaseException("Arithmetic not supported for this type");
    }
}

Value Value::Subtract(const Value& other) const {
    if (type_ != other.type_) {
        throw DatabaseException("Type mismatch in arithmetic");
    }
    switch (type_) {
        case TypeId::INTEGER: 
            return Value(integer_ - other.integer_);
        case TypeId::BIGINT:  
            return Value(bigint_ - other.bigint_);
        case TypeId::DECIMAL: 
            return Value(decimal_ - other.decimal_);
        default:
            throw DatabaseException("Arithmetic not supported for this type");
    }
}

Value Value::Multiply(const Value& other) const {
    if (type_ != other.type_) {
        throw DatabaseException("Type mismatch in arithmetic");
    }
    switch (type_) {
        case TypeId::INTEGER: 
            return Value(integer_ * other.integer_);
        case TypeId::BIGINT:  
            return Value(bigint_ * other.bigint_);
        case TypeId::DECIMAL:
            return Value(decimal_ * other.decimal_);
        default:
            throw DatabaseException("Arithmetic not supported for this type");
    }
}

Value Value::Divide(const Value& other) const {
    if (type_ != other.type_) {
        throw DatabaseException("Type mismatch in arithmetic");
    }
    switch (type_) {
        case TypeId::INTEGER:
            if (other.integer_ == 0) throw DatabaseException("Division by zero");
            return Value(integer_ / other.integer_);
        case TypeId::BIGINT:
            if (other.bigint_ == 0) throw DatabaseException("Division by zero");
            return Value(bigint_ / other.bigint_);
        case TypeId::DECIMAL:
            if (other.decimal_ == 0.0) return Value(std::numeric_limits<double>::infinity());
            return Value(decimal_ / other.decimal_);
        default:
            throw DatabaseException("Arithmetic not supported for this type");
    }
}



std::string Value::ToString() const {
    switch (type_) {
        case TypeId::INTEGER:
            return std::to_string(integer_);
        case TypeId::BIGINT:
            return std::to_string(bigint_);

        case TypeId::DECIMAL: {

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << decimal_;
            std::string s = oss.str();

            auto dot = s.find('.');
            if (dot != std::string::npos) {
                auto last_nonzero = s.find_last_not_of('0');

                if (last_nonzero == dot) {
                    s.erase(last_nonzero + 2);  
                } else {
                    s.erase(last_nonzero + 1);
                }
            }
            return s;
        }
        case TypeId::DATE: {
            int year, month, day;
            DaysToDate(date_, year, month, day);
            std::ostringstream oss;

            oss << std::setfill('0') << std::setw(4) << year
                << '-' << std::setw(2) << month
                << '-' << std::setw(2) << day;

            return oss.str();
        }
        case TypeId::VARCHAR:
            return varchar_;
        case TypeId::INVALID:
            return "NULL";
    }
    return "NULL";
}



Value Value::FromString(TypeId type, const std::string& str) {
    switch (type) {
        case TypeId::INTEGER:
            return Value(static_cast<int32_t>(std::stoi(str)));
        case TypeId::BIGINT:
            return Value(static_cast<int64_t>(std::stoll(str)));
        case TypeId::DECIMAL:
            return Value(std::stod(str));
        case TypeId::VARCHAR:
            return Value(str);
        case TypeId::DATE: { //parse YYYY-MM-DD format
            int year = std::stoi(str.substr(0, 4));
            int month = std::stoi(str.substr(5, 2));
            int day = std::stoi(str.substr(8, 2));
            return MakeDate(DateToDays(year, month, day));
        }
        case TypeId::INVALID:
            throw DatabaseException("Cannot parse INVALID type from string");
    }
    throw DatabaseException("Unknown TypeId in FromString");
}



size_t Value::Hash() const {
    switch (type_) {
        case TypeId::INTEGER:
            return std::hash<uint8_t>{}(static_cast<uint8_t>(type_)) ^ std::hash<int32_t>{}(integer_);
        case TypeId::BIGINT:
            return std::hash<uint8_t>{}(static_cast<uint8_t>(type_)) ^ std::hash<int64_t>{}(bigint_);
        case TypeId::DECIMAL:
            return std::hash<uint8_t>{}(static_cast<uint8_t>(type_)) ^ std::hash<double>{}(decimal_);
        case TypeId::VARCHAR:
            return std::hash<uint8_t>{}(static_cast<uint8_t>(type_)) ^ std::hash<std::string>{}(varchar_);
        case TypeId::DATE:
            return std::hash<uint8_t>{}(static_cast<uint8_t>(type_)) ^ std::hash<int32_t>{}(date_);
        case TypeId::INVALID:
            return 0;
    }
    return 0;
}



uint32_t Value::GetFixedLength() const {
    switch (type_) {
        case TypeId::INTEGER: 
            return 4;
        case TypeId::BIGINT:  
            return 8;
        case TypeId::DECIMAL: 
            return 8;
        case TypeId::DATE:    
            return 4;
        case TypeId::VARCHAR: 
            return 0;
        default:              
            return 0;
    }
}


bool Value::IsNull() const { return type_ == TypeId::INVALID; }

}  // namespace shilmandb
