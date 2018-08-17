#ifndef __EDGES_SUBPIX_H__
#define __EDGES_SUBPIX_H__
#include <opencv2/opencv.hpp>
#include <vector>


namespace sp
{
	class CV_EXPORTS EdgesSubPix
	{
	public:
		struct Contour
		{
			std::vector<cv::Point2f> points;
			std::vector<cv::Point> pointsInPix;
			std::vector<float> direction;
			std::vector<float> response;
			cv::Vec4i hierarchy;
            std::vector<float> nx;
            std::vector<float> ny;
			double length = 0.0;
			double area = 0.0;

			Contour() {}
			Contour(const Contour& contour)
			{
				points = contour.points;
				pointsInPix = contour.pointsInPix;
				direction = contour.direction;
				response = contour.response;
                nx = contour.nx;
                ny = contour.ny;
				length = contour.length;
				area = contour.area;
			}
			Contour(const std::vector<cv::Point2f>& _points, const std::vector<cv::Point>& _pointsInPix, const std::vector<float>& _direction, const std::vector<float>& _response, const cv::Vec4i& _hierarchy, const std::vector<float>& _nx, const std::vector<float>& _ny, const double& _length = 0.0, const double& _area = 0.0)
			{
				points = _points;
				pointsInPix = _pointsInPix;
				direction = _direction;
				response = _response;
				nx = _nx;
				ny = _ny;
				length = _length;
				area = _area;
			}
		};

		struct Edge
		{
			cv::Point2f point;
			cv::Point2f pointInPix;
			float direction;
			float response;
            float nx;
            float ny;

			Edge() {}
			Edge(const Edge& edge)
			{
				point = edge.point;
				pointInPix = edge.pointInPix;
				direction = edge.direction;
				response = edge.response;
                nx = edge.nx;
                ny = edge.ny;
			}
			Edge(const cv::Point2f& _point, const cv::Point& _pointInPix, const float& _direction, const float& _response, const cv::Vec4i& _hierarchy, const float _nx, const float& _ny)
			{
				point = _point;
				pointInPix = _pointInPix;
				direction = _direction;
				response = _response;
				nx = _nx;
				ny = _ny;
			}
		};

		static const double UNDEFINED_RESPONSE;
		static const double UNDEFINED_DIRECTION;

		EdgesSubPix();
		~EdgesSubPix();

		// only 8-bit (fusing the two lists (pix and not pix: TODO))
		void edgesSubPix(const cv::Mat &gray, double alpha, int low, int high, const cv::Mat& mask, std::vector<Edge>& edgesInPixels, std::vector<Edge>& edgesInSubPixel, cv::Mat& edges);
		void contoursSubPix(const cv::Mat &gray, double alpha, int low, int high, const cv::Mat& mask, std::vector<Edge>& edgesInPixel, std::vector<Contour> &contoursInPixel,	std::vector<Contour> &contoursInSubPixel, int mode, cv::Mat& egdes);
	};
}

#endif // __EDGES_SUBPIX_H__
