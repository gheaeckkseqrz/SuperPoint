#pragma once

template <typename T>
struct Vec2_
{
  Vec2_(int x_ = 0, int y_ = 0)
    : x(x_)
    , y(y_)
  {
  }

  T x = 0;
  T y = 0;
};

using Vec2 = Vec2_<int>;
