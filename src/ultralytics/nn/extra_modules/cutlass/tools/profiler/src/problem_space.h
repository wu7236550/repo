/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdlib>

#include "cutlass/util/command_line.h"

#include "cutlass/library/library.h"

#include "enumerated_types.h"

namespace cutlass {
namespace profiler {


struct ArgumentDescription {
    ArgumentTypeID type;

    std::vector<std::string> aliases;

    std::string description;


    ArgumentDescription() : type(ArgumentTypeID::kInvalid) {}

    ArgumentDescription(ArgumentTypeID type_,
                        std::vector<std::string> const& aliases_,
                        std::string const& description_)
            : type(type_), aliases(aliases_), description(description_) {}
};

using ArgumentDescriptionVector = std::vector<ArgumentDescription>;


struct KernelArgument {

    struct Value {
        KernelArgument const* argument;
        bool not_null;


        Value(KernelArgument const* argument_ = nullptr, bool not_null_ = true)
                : argument(argument_), not_null(not_null_) {}

        virtual ~Value() {}

        virtual std::ostream& print(std::ostream& out) const = 0;
    };

    struct ValueIterator {
        KernelArgument const* argument;

        bool null_argument;


        ValueIterator(KernelArgument const* argument_ = nullptr,
                      bool null_argument_ = false)
                : argument(argument_), null_argument(null_argument_) {
            if (!argument_->not_null()) {
                null_argument = true;
            }
        }

        virtual ~ValueIterator() {}

        virtual void operator++() = 0;

        virtual bool operator==(ValueIterator const& it) const = 0;

        virtual std::unique_ptr<Value> at() const = 0;

        ArgumentTypeID type() const { return argument->description->type; }

        bool operator!=(ValueIterator const& it) const {
            return !(*this == it);
        }

        std::ostream& print(std::ostream& out) const;
    };


    ArgumentDescription const* description;

    KernelArgument* parent;

    int ordinal;


    KernelArgument(ArgumentDescription const* description_ = nullptr,
                   KernelArgument* parent_ = nullptr, int ordinal_ = -1)
            : description(description_), parent(parent_), ordinal(ordinal_) {}

    virtual ~KernelArgument();

    virtual bool not_null() const = 0;

    std::string qualified_name() const {
        if (description) {
            if (description->aliases.empty()) {
                return "<description_not_null_no_aliases>";
            }
            return description->aliases.front();
        }
        return "<description_null>";
    }

    virtual std::unique_ptr<ValueIterator> begin() const = 0;
    virtual std::unique_ptr<ValueIterator> end() const = 0;
};

using KernelArgumentVector = std::vector<std::unique_ptr<KernelArgument>>;


struct ScalarArgument : public KernelArgument {

    struct ScalarValue : public KernelArgument::Value {
        std::string value;


        ScalarValue(std::string const& value_ = "",
                    ScalarArgument const* argument = nullptr,
                    bool not_null_ = true);

        virtual std::ostream& print(std::ostream& out) const;
    };

    using ValueCollection = std::vector<std::string>;

    struct ScalarValueIterator : public KernelArgument::ValueIterator {

        ValueCollection::const_iterator value_it;


        ScalarValueIterator(ScalarArgument const* argument = nullptr);

        virtual void operator++();
        virtual bool operator==(ValueIterator const& it) const;

        virtual std::unique_ptr<KernelArgument::Value> at() const;
    };


    ValueCollection values;


    ScalarArgument(ArgumentDescription const* description)
            : KernelArgument(description) {}

    virtual bool not_null() const { return !values.empty(); }

    virtual std::unique_ptr<KernelArgument::ValueIterator> begin() const;
    virtual std::unique_ptr<KernelArgument::ValueIterator> end() const;
};


struct Range {

    enum class Mode { kSequence, kRandom, kRandomLog2, kInvalid };

    struct Iterator {
        int64_t value;
        int64_t increment;
        Range const* range;


        Iterator(int64_t value_ = 0, int64_t increment_ = 1,
                 Range const* range_ = nullptr)
                : value(value_), increment(increment_), range(range_) {}

        Iterator& operator++() {
            value += increment;
            return *this;
        }

        Iterator operator++(int) {
            Iterator self(*this);
            ++(*this);
            return self;
        }

        bool operator==(Iterator const& it) const { return value == it.value; }

        bool operator!=(Iterator const& it) const { return !(*this == it); }

        static int64_t round(int64_t value, int64_t divisible) {
            int64_t rem = (value % divisible);

            if (rem > divisible / 2) {
                value += (divisible - rem);
            } else {
                value -= rem;
            }

            return value;
        }

        int64_t at() const {
            if (!range) {
                return value;
            }

            switch (range->mode) {
                case Mode::kSequence:
                    return value;

                case Mode::kRandom: {
                    double rnd = double(range->minimum) +
                                 double(std::rand()) / double(RAND_MAX) *
                                         (double(range->maximum) -
                                          double(range->minimum));

                    int64_t value = int64_t(rnd);

                    return round(value, range->divisible);
                } break;

                case Mode::kRandomLog2: {
                    double lg2_minimum =
                            std::log(double(range->minimum)) / std::log(2.0);
                    double lg2_maximum =
                            std::log(double(range->maximum)) / std::log(2.0);
                    double rnd = lg2_minimum +
                                 double(std::rand()) / double(RAND_MAX) *
                                         (lg2_maximum - lg2_minimum);

                    int64_t value = int64_t(std::pow(2.0, rnd));

                    return round(value, range->divisible);
                } break;
                default:
                    break;
            }
            return value;
        }

        int64_t operator*() const { return at(); }
    };


    int64_t first;
    int64_t last;
    int64_t increment;

    Mode mode;
    int64_t minimum;
    int64_t maximum;
    int64_t divisible;


    Range(int64_t first_ = 0)
            : first(first_),
              last(first_),
              increment(1),
              mode(Mode::kSequence),
              minimum(0),
              maximum(0),
              divisible(1) {}

    Range(int64_t first_, int64_t last_, int64_t increment_ = 1,
          Mode mode_ = Mode::kSequence, int64_t minimum_ = 0,
          int64_t maximum_ = 0, int64_t divisible_ = 1)
            : first(first_),
              last(last_),
              increment(increment_),
              mode(mode_),
              minimum(minimum_),
              maximum(maximum_),
              divisible(divisible_) {
        if (increment > 0) {
            if (last < first) {
                std::swap(last, first);
            }
        } else if (increment < 0) {
            if (first < last) {
                std::swap(last, first);
            }
        } else if (last != first) {
            last = first;
            increment = 1;
        }
    }

    static Range Sequence(int64_t first_, int64_t last_,
                          int64_t increment_ = 1) {
        return Range(first_, last_, increment_, Mode::kSequence);
    }

    static Range Random(int64_t minimum_, int64_t maximum_, int64_t count_,
                        int64_t divisible_ = 1) {
        return Range(1, count_, 1, Mode::kRandom, minimum_, maximum_,
                     divisible_);
    }

    static Range RandomLog2(int64_t minimum_, int64_t maximum_, int64_t count_,
                            int64_t divisible_ = 1) {
        return Range(1, count_, 1, Mode::kRandomLog2, minimum_, maximum_,
                     divisible_);
    }

    Iterator begin() const { return Iterator(first, increment, this); }

    Iterator end() const {
        return Iterator(first + ((last - first) / increment + 1) * increment,
                        increment, this);
    }
};

struct IntegerArgument : public KernelArgument {

    struct IntegerValue : public KernelArgument::Value {
        int64_t value;


        IntegerValue(int64_t value_ = 0,
                     IntegerArgument const* argument_ = nullptr,
                     bool not_null_ = true);

        virtual std::ostream& print(std::ostream& out) const;
    };

    using RangeCollection = std::vector<Range>;

    struct IntegerValueIterator : public KernelArgument::ValueIterator {

        RangeCollection::const_iterator range_it;
        Range::Iterator value_it;


        IntegerValueIterator();
        IntegerValueIterator(IntegerArgument const* argument);

        virtual void operator++();
        virtual bool operator==(ValueIterator const& it) const;

        virtual std::unique_ptr<KernelArgument::Value> at() const;
    };


    RangeCollection ranges;


    IntegerArgument(ArgumentDescription const* description)
            : KernelArgument(description) {}

    virtual bool not_null() const {
        bool _not_null = !ranges.empty();
        return _not_null;
    }

    virtual std::unique_ptr<KernelArgument::ValueIterator> begin() const;
    virtual std::unique_ptr<KernelArgument::ValueIterator> end() const;
};


struct TensorArgument : public KernelArgument {

    struct TensorDescription {
        library::NumericTypeID element;

        library::LayoutTypeID layout;

        std::vector<int> extent;

        std::vector<int> stride;


        TensorDescription(
                library::NumericTypeID element_ =
                        library::NumericTypeID::kUnknown,
                library::LayoutTypeID layout_ = library::LayoutTypeID::kUnknown,
                std::vector<int> extent_ = std::vector<int>(),
                std::vector<int> stride_ = std::vector<int>())
                : element(element_),
                  layout(layout_),
                  extent(extent_),
                  stride(stride_) {}
    };

    using ValueCollection = std::vector<TensorDescription>;

    struct TensorValue : public KernelArgument::Value {
        TensorDescription desc;


        TensorValue(TensorDescription const& desc_ = TensorDescription(),
                    TensorArgument const* argument_ = nullptr,
                    bool not_null_ = true);

        virtual std::ostream& print(std::ostream& out) const;
    };

    struct TensorValueIterator : public KernelArgument::ValueIterator {

        ValueCollection::const_iterator value_it;


        TensorValueIterator(TensorArgument const* argument_);

        virtual void operator++();
        virtual bool operator==(ValueIterator const& it) const;

        virtual std::unique_ptr<KernelArgument::Value> at() const;
    };

    ValueCollection values;


    TensorArgument(ArgumentDescription const* description)
            : KernelArgument(description) {}

    virtual bool not_null() const { return !values.empty(); }

    virtual std::unique_ptr<KernelArgument::ValueIterator> begin() const;
    virtual std::unique_ptr<KernelArgument::ValueIterator> end() const;
};


struct EnumeratedTypeArgument : public KernelArgument {

    struct EnumeratedTypeValue : public KernelArgument::Value {
        std::string element;


        EnumeratedTypeValue(std::string const& element_ = std::string(),
                            EnumeratedTypeArgument const* argument_ = nullptr,
                            bool not_null_ = true);

        virtual std::ostream& print(std::ostream& out) const;
    };

    using ValueCollection = std::vector<std::string>;

    struct EnumeratedTypeValueIterator : public KernelArgument::ValueIterator {

        ValueCollection::const_iterator value_it;


        EnumeratedTypeValueIterator(
                EnumeratedTypeArgument const* argument_ = nullptr);

        virtual void operator++();
        virtual bool operator==(ValueIterator const& it) const;

        virtual std::unique_ptr<KernelArgument::Value> at() const;
    };


    ValueCollection values;


    EnumeratedTypeArgument(ArgumentDescription const* description)
            : KernelArgument(description) {}

    virtual bool not_null() const { return !values.empty(); }

    virtual std::unique_ptr<KernelArgument::ValueIterator> begin() const;
    virtual std::unique_ptr<KernelArgument::ValueIterator> end() const;
};


class ProblemSpace {
public:
    using Problem = std::vector<std::unique_ptr<KernelArgument::Value>>;

    using IteratorVector =
            std::vector<std::unique_ptr<KernelArgument::ValueIterator>>;

    class Iterator {
    private:
        IteratorVector iterators;

    public:

        explicit Iterator();
        Iterator(ProblemSpace const& problem_space);
        Iterator(Iterator&& it);

        Iterator(Iterator const&) = delete;
        Iterator& operator=(Iterator const& it) = delete;
        ~Iterator() = default;

        void operator++();

        Problem at() const;

        void move_to_end();

        bool operator==(Iterator const& it) const;

        bool operator!=(Iterator const& it) const { return !(*this == it); }

        Problem operator*() const { return at(); }

        std::ostream& print(std::ostream& out) const;

    private:
        void construct_(KernelArgument const* argument);
    };

public:

    KernelArgumentVector arguments;

    std::unordered_map<std::string, size_t> argument_index_map;

public:

    ProblemSpace() {}

    ProblemSpace(ArgumentDescriptionVector const& schema,
                 CommandLine const& cmdline);

    Iterator begin()
            const;
    Iterator end()
            const;

    size_t argument_index(char const* name) const;

    std::vector<std::string> argument_names() const;

    size_t rank() const { return arguments.size(); }

private:
    void clone_(KernelArgumentVector& kernel_args,
                ArgumentDescription const* arg_desc);

    void parse_(KernelArgument* arg, CommandLine const& cmdline);
};


bool arg_as_int(int& int_value, KernelArgument::Value const* value_ptr);

bool arg_as_int(int64_t& int_value, KernelArgument::Value const* value_ptr);

bool arg_as_int(int& int_value, char const* name,
                ProblemSpace const& problem_space,
                ProblemSpace::Problem const& problem);

bool arg_as_int(int64_t& int_value, char const* name,
                ProblemSpace const& problem_space,
                ProblemSpace::Problem const& problem);

bool arg_as_NumericTypeID(library::NumericTypeID& numeric_type,
                          KernelArgument::Value const* value_ptr);

bool arg_as_NumericTypeID(library::NumericTypeID& numeric_type,
                          char const* name, ProblemSpace const& problem_space,
                          ProblemSpace::Problem const& problem);

bool arg_as_LayoutTypeID(library::LayoutTypeID& layout_type,
                         KernelArgument::Value const* value_ptr);

bool arg_as_LayoutTypeID(library::LayoutTypeID& layout_type, char const* name,
                         ProblemSpace const& problem_space,
                         ProblemSpace::Problem const& problem);

bool arg_as_OpcodeClassID(library::OpcodeClassID& opcode_class,
                          KernelArgument::Value const* value_ptr);

bool arg_as_OpcodeClassID(library::OpcodeClassID& opcode_class,
                          char const* name, ProblemSpace const& problem_space,
                          ProblemSpace::Problem const& problem);

bool arg_as_SplitKModeID(library::SplitKMode& split_k_mode,
                         KernelArgument::Value const* value_ptr);

bool arg_as_SplitKModeID(library::SplitKMode& split_k_mode, char const* name,
                         ProblemSpace const& problem_space,
                         ProblemSpace::Problem const& problem);

bool arg_as_ConvModeID(library::ConvModeID& conv_mode,
                       KernelArgument::Value const* value_ptr);

bool arg_as_ConvModeID(library::ConvModeID& conv_mode, char const* name,
                       ProblemSpace const& problem_space,
                       ProblemSpace::Problem const& problem);

bool arg_as_IteratorAlgorithmID(
        library::IteratorAlgorithmID& iterator_algorithm,
        KernelArgument::Value const* value_ptr);

bool arg_as_IteratorAlgorithmID(
        library::IteratorAlgorithmID& iterator_algorithm, char const* name,
        ProblemSpace const& problem_space,
        ProblemSpace::Problem const& problem);

bool arg_as_ProviderID(library::Provider& provider,
                       KernelArgument::Value const* value_ptr);

bool arg_as_ProviderID(library::Provider& provider, char const* name,
                       ProblemSpace const& problem_space,
                       ProblemSpace::Problem const& problem);

bool arg_as_scalar(std::vector<uint8_t>& bytes,
                   library::NumericTypeID numeric_type,
                   KernelArgument::Value const* value_ptr);

bool arg_as_scalar(std::vector<uint8_t>& bytes,
                   library::NumericTypeID numeric_type, char const* name,
                   ProblemSpace const& problem_space,
                   ProblemSpace::Problem const& problem);

bool tensor_description_satisfies(library::TensorDescription const& tensor_desc,
                                  TensorArgument::TensorValue const* value_ptr);

bool tensor_description_satisfies(library::TensorDescription const& tensor_desc,
                                  char const* name,
                                  ProblemSpace const& problem_space,
                                  ProblemSpace::Problem const& problem);

bool conv_kind_satisfies(
        library::ConvKind const& conv_kind,
        EnumeratedTypeArgument::EnumeratedTypeValue const* value_ptr);

bool conv_kind_satisfies(library::ConvKind const& conv_kind, char const* name,
                         ProblemSpace const& problem_space,
                         ProblemSpace::Problem const& problem);

bool iterator_algorithm_satisfies(
        library::IteratorAlgorithmID const& iterator_algorithm,
        EnumeratedTypeArgument::EnumeratedTypeValue const* value_ptr);

bool iterator_algorithm_satisfies(
        library::IteratorAlgorithmID const& iterator_algorithm,
        char const* name, ProblemSpace const& problem_space,
        ProblemSpace::Problem const& problem);


}
}

