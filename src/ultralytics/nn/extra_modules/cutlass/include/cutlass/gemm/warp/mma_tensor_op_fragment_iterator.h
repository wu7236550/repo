
#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_conversion.h"

namespace cutlass {
namespace gemm {
namespace warp {


template <
        typename Shape_,
        typename AccumulatorShape_,
        int KBlocksColumn_,
        typename ElementAccumulator_,
        typename Element_,
        typename Layout_,
        typename InstructionShape_,
        typename OutputOp_,
        bool IsBetaZero_>
class MmaTensorOpFragmentIterator;


template <
        typename Shape_,
        typename AccumulatorShape_,
        int KBlocksColumn_,
        typename Element_,
        typename InstructionShape_,
        typename OutputOp_>
class MmaTensorOpFragmentIterator<
        Shape_, AccumulatorShape_, KBlocksColumn_, Element_, Element_,
        cutlass::layout::ColumnMajor, InstructionShape_, OutputOp_, true> {
public:
    using Shape = Shape_;

    using AccumulatorShape = AccumulatorShape_;

    static int const kKBlockColumn = KBlocksColumn_;

    using Element = Element_;

    using Layout = cutlass::layout::ColumnMajor;

    using InstructionShape = InstructionShape_;

    using OutputOp = OutputOp_;

    static bool const IsBetaZero = true;

    static int const kThreads = 32;

    struct Policy {
        static_assert(
                !(Shape::kRow % InstructionShape::kM) &&
                        !(Shape::kColumn % InstructionShape::kN),
                "Shape of warp-level Mma must be divisible by operator shape.");
        static_assert(
                !(AccumulatorShape::kRow % Shape::kRow) &&
                        !(AccumulatorShape::kColumn % Shape::kColumn),
                "Shape of Warp Accumulator must be divisible by warp shape.");
        static_assert(!(kKBlockColumn % Shape::kColumn),
                      "KBlock size must be divisible by warp shape.");

        static int const kIterations = AccumulatorShape::kCount / Shape::kCount;
    };

private:
    static int const kElementsPerAccess =
            InstructionShape::kM * InstructionShape::kN / kThreads;

    using MmaIterations = MatrixShape<Shape::kRow / InstructionShape::kM,
                                      Shape::kColumn / InstructionShape::kN>;
    using AccumulatorIterations =
            MatrixShape<AccumulatorShape::kRow / InstructionShape::kM,
                        AccumulatorShape::kColumn / InstructionShape::kN>;

    static int const kKBlockIterations =
            (AccumulatorShape::kColumn + kKBlockColumn - 1) / kKBlockColumn;
    static int const kResidualColumn =
            AccumulatorShape::kColumn - (kKBlockIterations - 1) * kKBlockColumn;
    static int const kKBlockColumnIterations =
            kKBlockColumn / Shape::kColumn *
            (AccumulatorShape::kRow / Shape::kRow);
    static int const kResidualIndex = kResidualColumn / Shape::kColumn *
                                      (AccumulatorShape::kRow / Shape::kRow);

public:

    using Fragment = Array<Element, Shape::kCount / kThreads>;

    using AccumulatorFragment =
            Array<Element, AccumulatorShape::kCount / kThreads>;

private:
    using AccessType = Array<Element, kElementsPerAccess>;

private:

    AccessType const* accumulators_;

    int index_;

    bool is_residual_tile_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpFragmentIterator(AccumulatorFragment const& accum)
            : accumulators_(reinterpret_cast<AccessType const*>(&accum)),
              index_(0),
              is_residual_tile_(true) {}

    CUTLASS_HOST_DEVICE
    void add_offset(int index_offset) {
        index_ += index_offset;
        if (is_residual_tile_ && index_ >= kKBlockColumnIterations) {
            index_ = index_ - kKBlockColumnIterations + kResidualIndex;
            is_residual_tile_ = false;
        }
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpFragmentIterator& operator++() {
        add_offset(1);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpFragmentIterator& operator--() {
        add_offset(-1);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag, OutputOp output_op) const {
        if (output_op.is_source_needed())
            assert(0);

        AccessType src_fragment;
        src_fragment.clear();

        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        int index_m =
                (index_ * MmaIterations::kRow) % AccumulatorIterations::kRow;
        int index_n = (index_ * MmaIterations::kRow) /
                      AccumulatorIterations::kRow * MmaIterations::kColumn;

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < MmaIterations::kColumn; n++) {
            for (int m = 0; m < MmaIterations::kRow; m++) {
                int accumulator_access_offset =
                        (n + index_n) * AccumulatorIterations::kRow + m +
                        index_m;

                frag_ptr[n * MmaIterations::kRow + m].clear();
                if (!(is_residual_tile_ && index_ >= kResidualIndex))
                    frag_ptr[n * MmaIterations::kRow + m] =
                            output_op(accumulators_[accumulator_access_offset],
                                      src_fragment);
            }
        }
    }
};


template <
        typename Shape_,
        typename AccumulatorShape_,
        int KBlocksColumn_,
        typename ElementAccumulator_,
        typename Element_,
        typename InstructionShape_,
        typename OutputOp_>
class MmaTensorOpFragmentIterator<Shape_, AccumulatorShape_, KBlocksColumn_,
                                  ElementAccumulator_, Element_,
                                  cutlass::layout::RowMajor, InstructionShape_,
                                  OutputOp_, true> {
public:
    using Shape = Shape_;

    using AccumulatorShape = AccumulatorShape_;

    static int const kKBlockColumn = KBlocksColumn_;

    using ElementAccumulator = ElementAccumulator_;

    using Element = Element_;

    using Layout = cutlass::layout::RowMajor;

    using InstructionShape = InstructionShape_;

    using OutputOp = OutputOp_;

    static bool const IsBetaZero = true;

    static int const kThreads = 32;

    struct Policy {
        static_assert(
                !(Shape::kRow % InstructionShape::kM) &&
                        !(Shape::kColumn % InstructionShape::kN),
                "Shape of warp-level Mma must be divisible by operator shape.");
        static_assert(
                AccumulatorShape::kRow == Shape::kRow,
                "Rows of Warp Accumulator must be the same as rows of warp");
        static_assert(
                !(AccumulatorShape::kColumn % Shape::kColumn),
                "Shape of Warp Accumulator must be divisible by warp shape.");
        static_assert(!(kKBlockColumn % Shape::kColumn),
                      "KBlock size must be divisible by warp shape.");

        static int const kIterations = AccumulatorShape::kCount / Shape::kCount;
    };

private:
    static int const kRowsPerIteration = 8;
    static int const kColumnsPerIteration = 16;
    static int const kElementsPerIteration =
            kRowsPerIteration * InstructionShape::kN / kThreads;
    static int const kElementsPerAccess =
            kRowsPerIteration * kColumnsPerIteration / kThreads;
    static int const kIterationsPerAccess =
            kElementsPerAccess / kElementsPerIteration;

    static int const kIterationsPerInstruction =
            InstructionShape::kM / kRowsPerIteration;

    static int const kAccessStride = kIterationsPerInstruction;

    using MmaIterations = MatrixShape<Shape::kRow / InstructionShape::kM,
                                      Shape::kColumn / InstructionShape::kN>;
    using AccumulatorIterations =
            MatrixShape<AccumulatorShape::kRow / InstructionShape::kM,
                        AccumulatorShape::kColumn / InstructionShape::kN>;

    using AccessIterations =
            MatrixShape<MmaIterations::kRow * kIterationsPerInstruction,
                        MmaIterations::kColumn / kIterationsPerAccess>;

    static int const kKBlockIterations =
            (AccumulatorShape::kColumn + kKBlockColumn - 1) / kKBlockColumn;
    static int const kResidualColumn =
            AccumulatorShape::kColumn - (kKBlockIterations - 1) * kKBlockColumn;
    static int const kKBlockColumnIterations = kKBlockColumn / Shape::kColumn;
    static int const kResidualIndex = kResidualColumn / Shape::kColumn;

public:

    using Fragment = Array<Element, Shape::kCount / kThreads>;

    using AccumulatorFragment =
            Array<ElementAccumulator, AccumulatorShape::kCount / kThreads>;

private:
    using AccessType = Array<ElementAccumulator, kElementsPerIteration>;
    using FragmentAccessType = Array<Element, kElementsPerIteration>;

private:

    AccessType const* accumulators_;

    int index_;

    bool is_residual_tile_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpFragmentIterator(AccumulatorFragment const& accum)
            : accumulators_(reinterpret_cast<AccessType const*>(&accum)),
              index_(0),
              is_residual_tile_(true) {}

    CUTLASS_HOST_DEVICE
    void add_offset(int index_offset) {
        index_ += index_offset;
        if (is_residual_tile_ && index_ >= kKBlockColumnIterations) {
            index_ = index_ - kKBlockColumnIterations + kResidualIndex;
            is_residual_tile_ = false;
        }
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpFragmentIterator& operator++() {
        add_offset(1);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpFragmentIterator& operator--() {
        add_offset(-1);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void set_index(int idx) { index_ = idx; }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag, OutputOp output_op) const {
        if (output_op.is_source_needed())
            assert(0);

        FragmentAccessType src_fragment;
        src_fragment.clear();

        FragmentAccessType* frag_ptr =
                reinterpret_cast<FragmentAccessType*>(&frag);

        int index = index_ * AccessIterations::kCount;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < AccessIterations::kCount; i++) {

            int accumulator_access_offset =
                    index / AccessIterations::kCount *
                            (MmaIterations::kColumn *
                             kIterationsPerInstruction) +
                    (index % AccessIterations::kCount) /
                            (AccessIterations::kColumn *
                             kIterationsPerInstruction) *
                            AccumulatorIterations::kColumn *
                            kIterationsPerInstruction +
                    (index %
                     (AccessIterations::kColumn * kIterationsPerInstruction)) /
                            kIterationsPerInstruction *
                            (kIterationsPerInstruction * kIterationsPerAccess) +
                    (index % kIterationsPerInstruction);
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kIterationsPerAccess; j++) {
                frag_ptr[i * kIterationsPerAccess + j].clear();
                if (!(is_residual_tile_ && index_ >= kResidualIndex))
                    frag_ptr[i * kIterationsPerAccess + j] =
                            output_op(accumulators_[accumulator_access_offset +
                                                    j * kAccessStride],
                                      src_fragment);
            }
            index++;
        }
    }
};


}
}
}

