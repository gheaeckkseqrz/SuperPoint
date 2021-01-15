#include "imageGenerator.h"

std::array<float, 3> generateRandomColor()
{
  std::array<float, 3> color;
  color[0] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 255;
  color[1] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 255;
  color[2] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 255;
  return color;
}

TrainingImage SyntheticShapeGenerator::generateTrainingImage(unsigned int w, unsigned int h) const
{
  TrainingImage img(w, h);
  generateBackground(img);

  unsigned int mode = rand() % 2;
  switch (mode)
  {
  case 0:
    drawTriangle(img);
    break;
  case 1:
    drawRectangle(img);
    break;
  default:
    break;
  }
  return img;
}

void SyntheticShapeGenerator::generateBackground(TrainingImage &img) const
{
  // Solid color fill
  std::array<float, 3> background_color = generateRandomColor();
  cimg_forXY(img._img, x, y)
  {
    img._img(x, y, 0, 0) = background_color[0];
    img._img(x, y, 0, 1) = background_color[1];
    img._img(x, y, 0, 2) = background_color[2];
  }

  // Draw N ellipses
  unsigned int nb_of_elipses = rand() % 11 + 2; // In range [2, 12]
  for (unsigned int i(0); i < nb_of_elipses; ++i)
  {
    std::array<float, 3> ellipse_color = generateRandomColor();
    int ellipse_center_x = rand() % img._img.width();
    int ellipse_center_y = rand() % img._img.height();
    int ellipse_radius_1 = rand() % img._img.width();
    int ellipse_radius_2 = rand() % img._img.height();
    float ellipse_angle = static_cast<float>(rand()) * 360.0f / static_cast<float>(RAND_MAX);
    img._img.draw_ellipse(ellipse_center_x, ellipse_center_y, ellipse_radius_1, ellipse_radius_2, ellipse_angle, ellipse_color.data());
  }

  // Blur
  img._img.blur(16.0f, 16.0f, 16.0f);
}

void SyntheticShapeGenerator::drawTriangle(TrainingImage &img) const
{
  std::array<float, 3> triangle_color = generateRandomColor();
  Vec2 a, b, c;
  a.x = rand() % img._img.width();
  a.y = rand() % img._img.height();
  b.x = rand() % img._img.width();
  b.y = rand() % img._img.height();
  c.x = rand() % img._img.width();
  c.y = rand() % img._img.height();
  img._img.draw_triangle(a.x, a.y, b.x, b.y, c.x, c.y, triangle_color.data());
  img._keypoints.push_back(a);
  img._keypoints.push_back(b);
  img._keypoints.push_back(c);
}

void SyntheticShapeGenerator::drawRectangle(TrainingImage &img) const
{
  std::array<float, 3> rectangle_color = generateRandomColor();
  Vec2 a, b;
  a.x = rand() % img._img.width();
  a.y = rand() % img._img.height();
  b.x = rand() % img._img.width();
  b.y = rand() % img._img.height();
  img._img.draw_rectangle(a.x, a.y, b.x, b.y, rectangle_color.data());
  img._keypoints.push_back(a);
  img._keypoints.push_back(b);
  img._keypoints.push_back({a.x, b.y});
  img._keypoints.push_back({b.x, a.y});
}

void SyntheticShapeGenerator::drawTriangles(TrainingImage &img) const
{
  unsigned int number_of_triangles = rand() % 3 + 1; // Range [1, 3]
  for (unsigned int i(0); i < number_of_triangles; ++i)
    drawTriangle(img);
}
