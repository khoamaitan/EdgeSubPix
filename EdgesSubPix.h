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
			std::vector<float> direction;
			std::vector<float> response;
            std::vector<float> nx;
            std::vector<float> ny;
			double area=0.0;

			Contour() {}
			Contour(const Contour& contour)
			{
				points = contour.points;
				direction = contour.direction;
				response = contour.response;
                nx = contour.nx;
                ny = contour.ny;
				area = contour.area;
			}
		};

		struct Edge
		{
			cv::Point2f point;
			float direction;
			float response;
            float nx;
            float ny;

			Edge() {}
			Edge(const Edge& edge)
			{
				point = edge.point;
				direction = edge.direction;
				response = edge.response;
                nx = edge.nx;
                ny = edge.ny;
			}
		};

		static const double UNDEFINED_RESPONSE;
		static const double UNDEFINED_DIRECTION;

		EdgesSubPix();
		~EdgesSubPix();

		// only 8-bit
		void edgesSubPix(const cv::Mat &gray, double alpha, int low, int high, const cv::Mat& mask, std::vector<Edge>& edgesInPixels, std::vector<Edge>& edgesInSubPixel, cv::Mat& edges);

		void edgesSubPix(const cv::Mat &gray, double alpha, int low, int high, const cv::Mat& mask, std::vector<Edge>& edgesInPixel, std::vector<Contour> &contoursInPixel,
			std::vector<Contour> &contoursInSubPixel, cv::OutputArray hierarchy,
			int mode, cv::Mat& egdes);
	};
}

#endif // __EDGES_SUBPIX_H__
