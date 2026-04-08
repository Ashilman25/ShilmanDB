#pragma once
#include <stdexcept>


namespace shilmandb {


class DatabaseException : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
};

class IOException : public DatabaseException {
    public:
        using DatabaseException::DatabaseException;
};

class OutOfMemoryException : public DatabaseException {
    public:
        using DatabaseException::DatabaseException;
};


} //namespace shilmandb