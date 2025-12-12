#pragma once

#include "core/core.h"
#include <cassert>
#include <stdint.h>
#include <type_traits>
#include <utility>
#include <curand_kernel.h>

template<typename TagT, TagT TagValue, typename T>
struct TaggedType
{
    using Type = T;
    static constexpr TagT tag = TagValue;
};

namespace detail
{

template<typename TagT, TagT Query, typename... Cases>
struct TaggedTypeFor;

template<typename TagT, TagT Query, typename CaseHead, typename... Tail>
struct TaggedTypeFor<TagT, Query, CaseHead, Tail...>
{
    using Type = std::conditional_t<
        Query == CaseHead::tag,
        typename CaseHead::Type,
        typename TaggedTypeFor<TagT, Query, Tail...>::Type>;
};

template<typename TagT, TagT Query>
struct TaggedTypeFor<TagT, Query>
{
    using Type = void;
};

template<typename CaseHead, typename... Tail>
struct TaggedFirstType
{
    using Type = typename CaseHead::Type;
};

template<typename TagT, typename... Cases>
struct TaggedDispatcher;

template<typename TagT, typename CaseHead, typename... Tail>
struct TaggedDispatcher<TagT, CaseHead, Tail...>
{
    template<typename F>
    QUAL_CPU_GPU static decltype(auto) Dispatch(TagT activeTag, const void* ptr, F&& func)
    {
        if (activeTag == CaseHead::tag)
        {
            return func(static_cast<const typename CaseHead::Type*>(ptr));
        }

        if constexpr (sizeof...(Tail) > 0)
        {
            return TaggedDispatcher<TagT, Tail...>::Dispatch(activeTag, ptr, std::forward<F>(func));
        }
        else
        {
#ifndef __CUDA_ARCH__
            assert(false && "Invalid TaggedPointer dispatch");
#endif
            using FallbackType = typename TaggedFirstType<CaseHead>::Type;
            return func(static_cast<const FallbackType*>(nullptr));
        }
    }
};

} // namespace detail

template<typename TagT, typename... Cases>
class TaggedPointer
{
public:
    static constexpr uint8_t kInvalid = 0xFF;

    constexpr TaggedPointer() = default;
    constexpr TaggedPointer(TagT tag, const void* ptr)
        : Type(static_cast<uint8_t>(tag)), Ptr(ptr) {}

    QUAL_CPU_GPU bool IsValid() const { return Type != kInvalid && Ptr != nullptr; }
    QUAL_CPU_GPU void Reset()
    {
        Type = kInvalid;
        Ptr = nullptr;
    }

    QUAL_CPU_GPU TagT Tag() const { return static_cast<TagT>(Type); }

    template<TagT QueryTag>
    QUAL_CPU_GPU const auto* As() const
    {
        using TargetType = typename detail::TaggedTypeFor<TagT, QueryTag, Cases...>::Type;
        if constexpr (std::is_void_v<TargetType>)
        {
            return static_cast<const TargetType*>(nullptr);
        }
        else
        {
            return Tag() == QueryTag ? static_cast<const TargetType*>(Ptr) : nullptr;
        }
    }

    template<typename F>
    QUAL_CPU_GPU decltype(auto) dispatch(F&& func) const
    {
        return detail::TaggedDispatcher<TagT, Cases...>::Dispatch(Tag(), Ptr, std::forward<F>(func));
    }

    uint8_t Type = kInvalid;
    const void* Ptr = nullptr;
};

