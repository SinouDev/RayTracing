#pragma once

#include <memory>

template<typename T>
struct BaseObject {
public:
	virtual inline std::shared_ptr<T> Clone() const = 0;
};