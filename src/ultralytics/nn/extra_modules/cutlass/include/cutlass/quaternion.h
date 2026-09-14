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

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/matrix.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/vector.h"

namespace cutlass {


template <typename Element_ = float
          >
class Quaternion : public Array<Element_, 4> {
public:
    static int const kRank = 1;

    static int const kExtent = 4;

    using Base = Array<Element_, kExtent>;

    using Element = typename Base::Element;

    using Reference = typename Base::reference;

    using Index = int;

    static int const kX = 0;

    static int const kY = 1;

    static int const kZ = 2;

    static int const kW = 3;

public:

    CUTLASS_HOST_DEVICE
    Quaternion(Element w_ = Element(1)) {
        Base::at(kX) = Element(0);
        Base::at(kY) = Element(0);
        Base::at(kZ) = Element(0);
        Base::at(kW) = w_;
    }

    CUTLASS_HOST_DEVICE
    Quaternion(Element x_, Element y_, Element z_, Element w_) {
        Base::at(kX) = x_;
        Base::at(kY) = y_;
        Base::at(kZ) = z_;
        Base::at(kW) = w_;
    }

    CUTLASS_HOST_DEVICE
    Quaternion(Matrix3x1<Element> const& imag_, Element w_ = Element()) {
        Base::at(kX) = imag_[0];
        Base::at(kY) = imag_[1];
        Base::at(kZ) = imag_[2];
        Base::at(kW) = w_;
    }

    CUTLASS_HOST_DEVICE
    Reference at(Index idx) const { return Base::at(idx); }

    CUTLASS_HOST_DEVICE
    Reference at(Index idx) { return Base::at(idx); }

    CUTLASS_HOST_DEVICE
    Element x() const { return Base::at(kX); }

    CUTLASS_HOST_DEVICE
    Reference x() { return Base::at(kX); }

    CUTLASS_HOST_DEVICE
    Element y() const { return Base::at(kY); }

    CUTLASS_HOST_DEVICE
    Reference y() { return Base::at(kY); }

    CUTLASS_HOST_DEVICE
    Element z() const { return Base::at(kZ); }

    CUTLASS_HOST_DEVICE
    Reference z() { return Base::at(kZ); }

    CUTLASS_HOST_DEVICE
    Element w() const { return Base::at(kW); }

    CUTLASS_HOST_DEVICE
    Reference w() { return Base::at(kW); }

    CUTLASS_HOST_DEVICE
    Matrix3x1<Element> pure() const {
        return Matrix3x1<Element>(x(), y(), z());
    }

    CUTLASS_HOST_DEVICE
    static Quaternion<Element> rotation(
            Matrix3x1<Element> const& axis_unit,
            Element theta) {

        Element s = fast_sin(theta / Element(2));

        return Quaternion(s * axis_unit[0], s * axis_unit[1], s * axis_unit[2],
                          fast_cos(theta / Element(2)));
    }

    CUTLASS_HOST_DEVICE
    static Quaternion<Element> rotation(
            Element r_x, Element r_y, Element r_z,
            Element theta) {

        return rotation({r_x, r_y, r_z}, theta);
    }

    CUTLASS_HOST_DEVICE
    Matrix3x1<Element> rotate(Matrix3x1<Element> const& rhs) const {
        return (*this * Quaternion<Element>(rhs, 0) * reciprocal(*this)).pure();
    }

    CUTLASS_HOST_DEVICE
    Matrix3x1<Element> rotate_inv(Matrix3x1<Element> const& rhs) const {
        return (reciprocal(*this) * Quaternion<Element>(rhs, 0) * *this).pure();
    }

    CUTLASS_HOST_DEVICE
    Matrix3x1<Element> spinor(Matrix3x1<Element> const& rhs) const {
        return (*this * Quaternion<Element>(rhs, 0) * conj(*this)).pure();
    }

    CUTLASS_HOST_DEVICE
    Matrix3x1<Element> spinor_inv(Matrix3x1<Element> const& rhs) const {
        return (conj(*this) * Quaternion<Element>(rhs, 0) * *this).pure();
    }

    template <typename Element>
    CUTLASS_HOST_DEVICE Quaternion<Element>& operator+=(
            Quaternion<Element> const& rhs) {
        *this = (*this + rhs);
        return *this;
    }

    template <typename Element>
    CUTLASS_HOST_DEVICE Quaternion<Element>& operator-=(
            Quaternion<Element> const& rhs) {
        *this = (*this - rhs);
        return *this;
    }

    template <typename T>
    CUTLASS_HOST_DEVICE Quaternion<Element>& operator*=(
            Quaternion<Element> const& rhs) {
        *this = (*this * rhs);
        return *this;
    }

    template <typename T>
    CUTLASS_HOST_DEVICE Quaternion<Element>& operator*=(Element s) {
        *this = (*this * s);
        return *this;
    }

    template <typename T>
    CUTLASS_HOST_DEVICE Quaternion<Element>& operator/=(
            Quaternion<Element> const& rhs) {
        *this = (*this / rhs);
        return *this;
    }

    template <typename T>
    CUTLASS_HOST_DEVICE Quaternion<Element>& operator/=(Element s) {
        *this = (*this / s);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    Matrix3x3<Element> as_rotation_matrix_3x3() const {
        Matrix3x3<Element> m(
                w() * w() + x() * x() - y() * y() - z() * z(),
                2 * x() * y() - 2 * w() * z(), 2 * x() * z() + 2 * w() * y(),

                2 * x() * y() + 2 * w() * z(),
                w() * w() - x() * x() + y() * y() - z() * z(),
                2 * y() * z() - 2 * w() * x(),

                2 * x() * z() - 2 * w() * y(), 2 * y() * z() + 2 * w() * x(),
                w() * w() - x() * x() - y() * y() + z() * z());
        return m;
    }

    CUTLASS_HOST_DEVICE
    Matrix4x4<Element> as_rotation_matrix_4x4() const {
        Matrix4x4<Element> m = Matrix4x4<Element>::identity();
        m.set_slice_3x3(as_rotation_matrix_3x3());
        return m;
    }
};


template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> make_Quaternion(
        Element w) {

    return Quaternion<Element>(w);
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> make_Quaternion(
        Matrix3x1<Element> const& imag,
        Element w) {

    return Quaternion<Element>(imag, w);
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> make_QuaternionRotation(
        Matrix3x1<Element> const& axis_unit,
        Element w) {

    return Quaternion<Element>::rotation(axis_unit, w);
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> make_Quaternion(Element x, Element y,
                                                        Element z, Element w) {
    return Quaternion<Element>(x, y, z, w);
}


template <typename Element>
CUTLASS_HOST_DEVICE Element abs(Quaternion<Element> const& q) {
    return fast_sqrt(norm(q));
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> conj(Quaternion<Element> const& q) {
    return make_Quaternion(-q.x(), -q.y(), -q.z(), q.w());
}

template <typename Element>
CUTLASS_HOST_DEVICE Element norm(Quaternion<Element> const& q) {
    return q.x() * q.x() + q.y() * q.y() + q.z() * q.z() + q.w() * q.w();
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> reciprocal(
        Quaternion<Element> const& q) {
    Element nsq = norm(q);

    return make_Quaternion(-q.x() / nsq, -q.y() / nsq, -q.z() / nsq,
                           q.w() / nsq);
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> unit(Quaternion<Element> const& q) {
    Element rcp_mag = Element(1) / abs(q);

    return make_Quaternion(q.x() * rcp_mag, q.y() * rcp_mag, q.z() * rcp_mag,
                           q.w() * rcp_mag);
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> exp(Quaternion<Element> const& q) {
    Element exp_ = fast_exp(q.w());
    Element imag_norm =
            fast_sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
    Element sin_norm = fast_sin(imag_norm);

    return make_Quaternion(exp_ * q.x() * sin_norm / imag_norm,
                           exp_ * q.y() * sin_norm / imag_norm,
                           exp_ * q.z() * sin_norm / imag_norm,
                           exp_ * fast_cos(imag_norm));
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> log(Quaternion<Element> const& q) {
    Element v = fast_sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
    Element s = fast_acos(q.w() / abs(q)) / v;

    return make_Quaternion(q.x() * s, q.y() * s, q.z() * s, fast_log(q.w()));
}

template <typename Element>
CUTLASS_HOST_DEVICE Element
get_rotation_angle(Quaternion<Element> const& q_unit) {
    return fast_acos(q_unit.w()) * Element(2);
}

template <typename Element>
CUTLASS_HOST_DEVICE Matrix3x1<Element> get_rotation_axis(
        Quaternion<Element> const& q_unit) {
    return q_unit.pure().unit();
}


template <typename Element>
CUTLASS_HOST_DEVICE bool operator==(Quaternion<Element> const& lhs,
                                    Quaternion<Element> const& rhs) {
    return lhs.x() == rhs.x() && lhs.y() == rhs.y() && lhs.z() == rhs.z() &&
           lhs.w() == rhs.w();
}

template <typename Element>
CUTLASS_HOST_DEVICE bool operator!=(Quaternion<Element> const& lhs,
                                    Quaternion<Element> const& rhs) {
    return !(lhs == rhs);
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> operator*(Quaternion<Element> q,
                                                  Element s) {
    return make_Quaternion(q.x() * s, q.y() * s, q.z() * s, q.w() * s);
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> operator*(
        Element s, Quaternion<Element> const& q) {
    return make_Quaternion(s * q.x(), s * q.y(), s * q.z(), s * q.w());
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> operator/(Quaternion<Element> const& q,
                                                  Element s) {
    return make_Quaternion(q.x() / s, q.y() / s, q.z() / s, q.w() / s);
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> operator-(
        Quaternion<Element> const& q) {
    return make_Quaternion(-q.x(), -q.y(), -q.z(), -q.w());
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> operator+(
        Quaternion<Element> const& lhs, Quaternion<Element> const& rhs) {
    return make_Quaternion(lhs.x() + rhs.x(), lhs.y() + rhs.y(),
                           lhs.z() + rhs.z(), lhs.w() + rhs.w());
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> operator-(
        Quaternion<Element> const& lhs, Quaternion<Element> const& rhs) {
    return make_Quaternion(lhs.x() - rhs.x(), lhs.y() - rhs.y(),
                           lhs.z() - rhs.z(), lhs.w() - rhs.w());
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> operator*(
        Quaternion<Element> const& lhs, Quaternion<Element> const& rhs) {
    return make_Quaternion(lhs.w() * rhs.x() + rhs.w() * lhs.x() +
                                   lhs.y() * rhs.z() - lhs.z() * rhs.y(),
                           lhs.w() * rhs.y() + rhs.w() * lhs.y() +
                                   lhs.z() * rhs.x() - lhs.x() * rhs.z(),
                           lhs.w() * rhs.z() + rhs.w() * lhs.z() +
                                   lhs.x() * rhs.y() - lhs.y() * rhs.x(),
                           lhs.w() * rhs.w() - lhs.x() * rhs.x() -
                                   lhs.y() * rhs.y() - lhs.z() * rhs.z());
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> operator/(
        Quaternion<Element> const& lhs, Quaternion<Element> const& rhs) {
    return lhs * reciprocal(rhs);
}

template <typename Element>
CUTLASS_HOST_DEVICE Quaternion<Element> operator/(
        Element s, Quaternion<Element> const& q) {
    return s * reciprocal(q);
}

template <typename Element>
CUTLASS_HOST_DEVICE Matrix3x1<Element> spinor_rotation(
        Quaternion<Element> const& spinor,
        Matrix3x1<Element> const& rhs) {

    return (spinor * Quaternion<Element>(rhs, 0) * conj(spinor)).pure();
}

template <typename Element>
CUTLASS_HOST_DEVICE Matrix3x1<Element> spinor_rotation_inv(
        Quaternion<Element> const& spinor,
        Matrix3x1<Element> const& rhs) {

    return (conj(spinor) * Quaternion<Element>(rhs, 0) * spinor).pure();
}



template <typename Element>
std::ostream& operator<<(std::ostream& out, Quaternion<Element> const& q) {
    return out << q.w() << "+i" << q.x() << "+j" << q.y() << "+k" << q.z();
}


}

