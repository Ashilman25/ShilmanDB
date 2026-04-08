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