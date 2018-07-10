#include <iterator>
#include <math.h>
#include "EdgesSubPix.h"
#include "subpix.h"


#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif



sp::SubPix::SubPix()
{
}


sp::SubPix::~SubPix()
{
}

void sp::SubPix::resetExtraction()
{
	m_inputPoints.clear();
	m_rois.clear();
	m_roiImages.clear();

	m_edgesInPixel.clear();
	m_edgesInSubPixel.clear();
	m_contoursPtsInSubPixel.clear();
	m_contoursPtsInPixel.clear();

	m_roisEdgesPts.clear();
	m_roisContoursPts.clear();

	m_contourStep = 0;
}

void sp::SubPix::resetAmbiguityImages()
{
	m_imageListEdgesAmbiguities.clear();
	m_imageListContoursAmbiguities.clear();
}

void sp::SubPix::extractEdges(int threshold)
{
    // check image
    if (m_image.empty()) {
        return;
    }

    // set thresholds
    m_low = threshold;
    m_high = 3 * threshold;

    // set image ROIS
    setImageROIs(m_rois, m_image);

    // extract contour from ROI
    m_nbOfEdges = 0;
    for (auto roiImage : m_roiImages) {

        // get ROIs
        cv::Mat imageROI, edgesROI;
        getImageROI(roiImage.first, imageROI, edgesROI);

        int64 t0 = cv::getCPUTickCount();
        m_edgesSubPix.edgesSubPix(imageROI, m_alpha, m_low, m_high, m_mask, m_edgesInPixel, m_edgesInSubPixel, edgesROI);
        int64 t1 = cv::getCPUTickCount();

        cv::Rect roi = getROI(roiImage.first);

        if (m_display) {
            if (!roi.empty()) {
                std::cout << "ROI " << roiImage.first << " (" << roi.x << "," << roi.y << "," << roi.width << "x" << roi.height << ")" << " : " << "execution time is " << (t1 - t0) / (double)cv::getTickFrequency() << " seconds" << std::endl;
				std::cout << "using mask : " << !m_mask.empty() << std::endl;
                std::cout << "nb edges (subpixel) : " << m_edgesInSubPixel.size() << std::endl;
                std::cout << "nb edges (pixel) : " << m_edgesInPixel.size() << std::endl;
            }
        }

        m_roisEdgesPts[roiImage.first] = m_edgesInSubPixel;

        m_nbOfEdges += (int)m_edgesInSubPixel.size();
    }

	// update edges
	if (m_roisEdgesPts.size() > 1) {
		updateEdgesListFromROIs();
	}

    // show edges
    if (m_display) {

        // show edges
        show(m_edges, m_EDGES_WINDOW_NAME);
    }

    // remove ROIs
    m_rois.clear();
}

void sp::SubPix::extractContours(int threshold)
{
    // check image
    if (m_image.empty()) {
        return;
    }

    // set thresholds
    m_low = threshold;
    m_high = 3 * threshold;

    // set image ROIS
    setImageROIs(m_rois, m_image);

    // extract contour from ROI
    m_nbOfContours = 0;
    for (auto roiImage : m_roiImages) {

        // get ROIs
        cv::Mat imageROI, edgesROI;
        getImageROI(roiImage.first, imageROI, edgesROI);

        int64 t0 = cv::getCPUTickCount();
        m_edgesSubPix.contoursSubPix(imageROI, m_alpha, m_low, m_high, m_mask, m_edgesInPixel, m_contoursPtsInPixel, m_contoursPtsInSubPixel, m_contourMode, edgesROI);
        int64 t1 = cv::getCPUTickCount();

        if (m_display) {

            cv::Rect roi = getROI(roiImage.first);
            if (!roi.empty()) {
                std::cout << "ROI " << roiImage.first << " (" << roi.x << "," << roi.y << "," << roi.width << "x" << roi.height << ")" << " : " << "execution time is " << (t1 - t0) / (double)cv::getTickFrequency() << " seconds" << std::endl;
				std::cout << "using mask : " << !m_mask.empty() << std::endl;
                std::cout << "nb contours (subpixel) : " << m_contoursPtsInSubPixel.size() << std::endl;
                std::cout << "nb contours (pixel) : " << m_contoursPtsInPixel.size() << std::endl;
            }
        }

        m_roisContoursPts[roiImage.first] = m_contoursPtsInSubPixel;

        m_nbOfContours += (int)m_contoursPtsInSubPixel.size();
    }

	// update contours
	if (m_roisContoursPts.size() > 1) {
		updateContoursListFromROIs();
	}

    // show contours
    if (m_display) {

        // show edges
        show(m_edges, m_EDGES_WINDOW_NAME);

        // limitation due to openCV bad profiling
        if (m_nbOfContours < m_maxContours) {
            cv::cvtColor(m_image, m_contours, cv::COLOR_GRAY2RGB);

            for (auto roiImage : m_roiImages) {

                auto contoursPts = m_roisContoursPts.find(roiImage.first);
                showContours(contoursPts->second, m_contours, m_CONTOURS_WINDOW_NAME, m_markers, m_markerType, m_normals);
            }
        }
        else {
            std::cout << "Number of contours " << m_nbOfContours << ">" << m_maxContours << ": not displayed for time consuming reason." << std::endl;
        }
    }

    // remove ROIs
    m_rois.clear();
}

void sp::SubPix::selectContour(int threshold)
{
    // check image
    if (m_image.empty()) {
        return;
    }

    // clean contours view
    cv::cvtColor(m_image, m_contours, cv::COLOR_GRAY2BGR);

    // show it in red
    cv::Scalar color(0, 0, 255);

    // show contours
    int countoursNumber = 0;
	int lastContourId = 0;
    std::map< std::string, std::vector< sp::EdgesSubPix::Contour> >::const_iterator roiContoursPts;
    for (roiContoursPts = m_roisContoursPts.begin(); roiContoursPts != m_roisContoursPts.end(); ++roiContoursPts)
    {
		lastContourId = countoursNumber - 1;
        countoursNumber += (int)roiContoursPts->second.size();
        if (threshold < countoursNumber) {
            break;
        }
    }

    m_selectedContourId = (roiContoursPts == m_roisContoursPts.begin()) ? threshold : threshold - lastContourId - 1;

	m_selectedContour = m_contoursPtsInSubPixel[m_selectedContourId];
	m_selectedContourInPixel = m_contoursPtsInPixel[m_selectedContourId];

	if (m_display) {
		std::cout << "selected contour " << m_selectedContourId << std::endl;
		std::cout << "nb of points at pixel resolution : " << m_selectedContourInPixel.points.size() << std::endl;
		std::cout << "nb of points at subpixel resolution : " << m_selectedContour.points.size() << std::endl;
		std::cout << "contour length at pixel resolution : " << m_selectedContourInPixel.length << std::endl;
		std::cout << "contour length at subpixel resolution : " << m_selectedContour.length << std::endl;
		std::cout << "contour area at pixel resolution : " << m_selectedContourInPixel.area << std::endl;
		std::cout << "contour area at subpixel resolution : " << m_selectedContour.area << std::endl;
		std::cout << "contour hierarchy : " << m_selectedContour.hierarchy[0] << " , " << m_selectedContour.hierarchy[1] << " , " << m_selectedContour.hierarchy[2] << " , " << m_selectedContour.hierarchy[3] << std::endl;
	}

    showContour(m_selectedContourId, roiContoursPts->second, m_contours, m_CONTOURS_WINDOW_NAME, color, m_markers, m_markerType, m_normals, 10, 10, m_selectContourStepByStep);
}

void sp::SubPix::selectAreaContours(int threshold)
{
	// check image
	if (m_image.empty()) {
		return;
	}

	// clean contours view
	cv::cvtColor(m_image, m_contours, cv::COLOR_GRAY2BGR);

	// set area threshold
	m_area_threshold = threshold;

	// show contours
	cv::RNG rng;
	std::map< std::string, std::vector< sp::EdgesSubPix::Contour> >::const_iterator roiContoursPts;

	// create new edges output image
	m_edgesFromContours = cv::Mat::zeros(m_contours.size(), CV_8UC1);

	for (roiContoursPts = m_roisContoursPts.begin(); roiContoursPts != m_roisContoursPts.end(); ++roiContoursPts)
	{
		std::vector< sp::EdgesSubPix::Contour> roiSelectedContours;
		for (std::vector< sp::EdgesSubPix::Contour>::const_iterator iter = roiContoursPts->second.begin(); iter != roiContoursPts->second.end(); ++iter)
		{
			if (iter->area > threshold)
			{
				cv::Scalar color(rng.uniform(1, 254), rng.uniform(1, 254), rng.uniform(1, 254));
				showContour((int)(iter - roiContoursPts->second.begin()), roiContoursPts->second, m_contours, m_CONTOURS_WINDOW_NAME, color, m_markers, m_markerType);
				roiSelectedContours.push_back(*iter);
			}
		}
		
		// show equivalent edges
		showEdgesFromContours(roiSelectedContours, m_contours, m_edgesFromContours, m_EDGES_FROM_CONTOURS);
	}
}

void sp::SubPix::selectLengthContours(int threshold)
{
	// check image
	if (m_image.empty()) {
		return;
	}

	// clean contours view
	cv::cvtColor(m_image, m_contours, cv::COLOR_GRAY2BGR);

	// set length threshold
	m_length_threshold = threshold;

	// show contours
	cv::RNG rng;
	std::map< std::string, std::vector< sp::EdgesSubPix::Contour> >::const_iterator roiContoursPts;

	// create new edges output image
	m_edgesFromContours = cv::Mat::zeros(m_contours.size(), CV_8UC1);

	for (roiContoursPts = m_roisContoursPts.begin(); roiContoursPts != m_roisContoursPts.end(); ++roiContoursPts)
	{
		std::vector< sp::EdgesSubPix::Contour> roiSelectedContours;
		for (std::vector< sp::EdgesSubPix::Contour>::const_iterator iter = roiContoursPts->second.begin(); iter != roiContoursPts->second.end(); ++iter)
		{
			if (iter->length > threshold)
			{
				cv::Scalar color(rng.uniform(1, 254), rng.uniform(1, 254), rng.uniform(1, 254));
				showContour((int)(iter - roiContoursPts->second.begin()), roiContoursPts->second, m_contours, m_CONTOURS_WINDOW_NAME, color, m_markers, m_markerType);
				roiSelectedContours.push_back(*iter);
			}
		}

		// show equivalent edges
		showEdgesFromContours(roiSelectedContours, m_contours, m_edgesFromContours, m_EDGES_FROM_CONTOURS);
	}
}

void sp::SubPix::selectNbOfPtsContours(int threshold)
{
	// check image
	if (m_image.empty()) {
		return;
	}

	// clean contours view
	cv::cvtColor(m_image, m_contours, cv::COLOR_GRAY2BGR);

	// set length threshold
	m_nbOfPts_threshold = threshold;

	// show contours
	cv::RNG rng;
	std::map< std::string, std::vector< sp::EdgesSubPix::Contour> >::const_iterator roiContoursPts;

	// create new edges output image
	m_edgesFromContours = cv::Mat::zeros(m_contours.size(), CV_8UC1);

	for (roiContoursPts = m_roisContoursPts.begin(); roiContoursPts != m_roisContoursPts.end(); ++roiContoursPts)
	{
		std::vector< sp::EdgesSubPix::Contour> roiSelectedContours;
		for (std::vector< sp::EdgesSubPix::Contour>::const_iterator iter = roiContoursPts->second.begin(); iter != roiContoursPts->second.end(); ++iter)
		{
			if (iter->points.size() > threshold)
			{
				cv::Scalar color(rng.uniform(1, 254), rng.uniform(1, 254), rng.uniform(1, 254));
				showContour((int)(iter - roiContoursPts->second.begin()), roiContoursPts->second, m_contours, m_CONTOURS_WINDOW_NAME, color, m_markers, m_markerType);
				roiSelectedContours.push_back(*iter);
			}
		}

		// show equivalent edges
		showEdgesFromContours(roiSelectedContours, m_contours, m_edgesFromContours, m_EDGES_FROM_CONTOURS);
	}
}

void sp::SubPix::selectOrientedContoursParts(int threshold)
{
	// check image
	if (m_image.empty()) {
		return;
	}

	// clean contours view
	cv::cvtColor(m_image, m_contours, cv::COLOR_GRAY2BGR);

	// set length threshold
	m_orientationTolerance = threshold;

	// show contours
	cv::RNG rng;
	std::map< std::string, std::vector< sp::EdgesSubPix::Contour> >::const_iterator roiContoursPts;

	// create new edges output image
	m_edgesFromContours = cv::Mat::zeros(m_contours.size(), CV_8UC1);

	for (roiContoursPts = m_roisContoursPts.begin(); roiContoursPts != m_roisContoursPts.end(); ++roiContoursPts)
	{
		std::vector< sp::EdgesSubPix::Contour> contoursCloneInliers;
		contoursCloneInliers.resize(roiContoursPts->second.size());
		std::copy(roiContoursPts->second.begin(), roiContoursPts->second.end(), contoursCloneInliers.begin());
		std::vector< sp::EdgesSubPix::Contour> contoursCloneOutliers;
		contoursCloneOutliers.resize(roiContoursPts->second.size());
		std::copy(roiContoursPts->second.begin(), roiContoursPts->second.end(), contoursCloneOutliers.begin());

		for (int iter=0; iter < (int)contoursCloneInliers.size() ; ++iter)
		{
			sp::EdgesSubPix::Contour& contour = contoursCloneInliers[iter];
			std::vector< cv::Point2f > ptsInliers, ptsOutliers;
			std::vector<float> directionInliers, directionOutliers;
			std::vector<float> responseInliers, responseOutliers;
			cv::Vec4i hierarchyInliers, hierarchyOutliers;
			std::vector<float> nxInliers, nxOutliers;
			std::vector<float> nyInliers, nyOutliers;
			double lengthInliers, lengthOutliers;
			double areaInliers, areaOutliers;
			cv::Vec4f line = contourOrientationLine(contour.points);
			for (int i=0; i<(int)contour.points.size(); ++i)
			{
				if (orientationPtFilter(cv::Point2f(contour.nx[i], contour.ny[i]), cv::Point2f(line[0], line[1]), m_orientationTolerance, m_angle_ref))
				{
					ptsInliers.push_back(contour.points[i]);
					directionInliers.push_back(contour.direction[i]);
					responseInliers.push_back(contour.response[i]);
					nxInliers.push_back(contour.nx[i]);
					nyInliers.push_back(contour.ny[i]);
				}
				else
				{
					ptsOutliers.push_back(contour.points[i]);
					directionOutliers.push_back(contour.direction[i]);
					responseOutliers.push_back(contour.response[i]);
					nxOutliers.push_back(contour.nx[i]);
					nyOutliers.push_back(contour.ny[i]);
				}
			}
			hierarchyInliers = contour.hierarchy;
			lengthInliers = contour.length;
			areaInliers = contour.area;

			sp::EdgesSubPix::Contour filteredOutlierContour(ptsOutliers, directionOutliers, responseOutliers, hierarchyOutliers, nxOutliers, nyOutliers, lengthOutliers, areaOutliers);
			sp::EdgesSubPix::Contour& contourOutlier = contoursCloneOutliers[iter];
			contourOutlier = filteredOutlierContour;

			cv::Scalar color(rng.uniform(1, 254), rng.uniform(1, 254), rng.uniform(1, 254));
			showContour(iter, contoursCloneOutliers, m_contours, m_CONTOURS_WINDOW_NAME, color, true, cv::MARKER_CROSS, false, 1, 1, false, 2);

			sp::EdgesSubPix::Contour filteredInlierContour(ptsInliers, directionInliers, responseInliers, hierarchyInliers, nxInliers, nyInliers, lengthInliers, areaInliers);
			contour = filteredInlierContour;
		}

		// show equivalent edges
		showEdgesFromContours(contoursCloneInliers, m_contours, m_edgesFromContours, m_EDGES_FROM_CONTOURS);
	}
}

void sp::SubPix::filterContours(int threshold, void* type)
{
	// get threshold
	int* threshold_type = (int*)type;

	if (threshold_type) {

		// check image
		if (m_image.empty()) {
			return;
		}

		// clean contours view
		cv::cvtColor(m_image, m_contours, cv::COLOR_GRAY2BGR);

		// set threshold
		switch (*threshold_type)
		{
		case NB_OF_PTS:
			m_nbOfPts_threshold = threshold;
			break;
		case LENGTH:
			m_length_threshold = threshold;
			break;
		case AREA:
			m_area_threshold = threshold;
			break;
		case ORIENTATION:
			m_orientationTolerance = threshold;
			break;
		case NONE:
			break;
		default:
			*threshold_type = NONE;
			break;
		};

		// show contours
		cv::RNG rng;
		std::map< std::string, std::vector< sp::EdgesSubPix::Contour> >::const_iterator roiContoursPts;

		// create new edges output image
		m_edgesFromContours = cv::Mat::zeros(m_contours.size(), CV_8UC1);

		for (roiContoursPts = m_roisContoursPts.begin(); roiContoursPts != m_roisContoursPts.end(); ++roiContoursPts)
		{
			std::vector< sp::EdgesSubPix::Contour> roiSelectedContours;
			roiSelectedContours.resize(roiContoursPts->second.size());
			std::copy(roiContoursPts->second.begin(), roiContoursPts->second.end(), roiSelectedContours.begin());
		
			if (*threshold_type != NONE) {
				filterContours(roiSelectedContours, m_CONTOURS_ORIENTATIONS);
			}
			cv::Scalar color(rng.uniform(1, 254), rng.uniform(1, 254), rng.uniform(1, 254));
			showContours(roiSelectedContours, m_contours, m_FILTERED_CONTOURS_WINDOW_NAME, true, cv::MARKER_CROSS, false, 1, 1, 1);

			// show equivalent edges
			showEdgesFromContours(roiSelectedContours, m_contours, m_edgesFromContours, m_EDGES_FROM_CONTOURS);

			// set selected roi contours
			m_roisContoursPtsFiltered[roiContoursPts->first] = roiSelectedContours;
		}
	}
}

void sp::SubPix::setEdgesFromContours(const std::vector< sp::EdgesSubPix::Contour> & contoursPts, const cv::Mat& contours, cv::Mat& edges)
{
	// clean contours view
	if (!contours.empty()) {

		if (edges.empty()) {
			edges = cv::Mat::zeros(contours.size(), CV_8UC1);
		}

		for (int i = 0; i < (int)contoursPts.size(); i++)
		{
			sp::EdgesSubPix::Contour contour = contoursPts[i];
			for (int j = 0; j < (int)contour.points.size(); j++)
			{
				cv::Point2f pt = contour.points[j];
				edges.at<uchar>((int)pt.y, (int)pt.x) = (uchar)255;
			}
		}
	}
}

void sp::SubPix::showEdgesFromContours(const std::vector< sp::EdgesSubPix::Contour> & contoursPts, const cv::Mat& contours, cv::Mat& edges, const std::string& windowName)
{
	// clean contours view
	setEdgesFromContours(contoursPts, contours, edges);
	show(edges, windowName);
}

bool sp::SubPix::nbPtsFilter(const sp::EdgesSubPix::Contour& contour, double threshold)
{
	if (contour.points.size() < threshold)
	{
		return false;
	}
	return true;
}
bool sp::SubPix::lengthFilter(const sp::EdgesSubPix::Contour& contour, double threshold)
{
	if (contour.length < threshold)
	{
		return false;
	}
	return true;
}
bool sp::SubPix::areaFilter(const sp::EdgesSubPix::Contour& contour, double threshold)
{
	if (contour.area < threshold)
	{
		return false;
	}
	return true;
}
bool sp::SubPix::orientationPtFilter(const cv::Point2f& pt, cv::Point2f lineDir, double orientationTolerance, double angle_ref)
{
	double dot = pt.dot(lineDir);
	dot /= cv::norm(pt)*cv::norm(lineDir);
	double angle = std::acos(dot);
	if (angle > M_PI / 2) {
		angle -= M_PI;
	}
	double diff = std::abs(angle - angle_ref);
	if (diff < orientationTolerance*M_PI / 180)
	{
		return false;
	}

	return true;
}

cv::Vec4f sp::SubPix::contourOrientationLine(const std::vector< cv::Point2f >& pts)
{
	cv::Vec4f line;
	cv::fitLine(pts, line, cv::DIST_L2, 0, 0.01, 0.01);
	return line;
}

void sp::SubPix::filterContours(std::vector<sp::EdgesSubPix::Contour>& contours, const std::string& windowName)
{
	cv::Mat output;
	cv::cvtColor(m_image, output, cv::COLOR_GRAY2BGR);
	cv::RNG rng;

	bool globalFiltering = m_filterNbOfPoints | m_filterLength;
	bool ptFiltering = m_filterOrientation | m_otherFilter;

	std::vector<sp::EdgesSubPix::Contour> validContours;

	for (int i = 0; i < (int) contours.size(); ++i)
	{	
		sp::EdgesSubPix::Contour& contour = contours[i];

		// global filtering
		if (globalFiltering)
		{
			bool nbOfPointsFlag=true;
			bool lengthFlag=true;
			bool areaFlag=true;

			// contour nb of points
			if (m_filterNbOfPoints)
			{
				if (!nbPtsFilter(contour, m_nbOfPts_threshold)) {
					nbOfPointsFlag = false;
					continue;
				}
			}

			// contour length
			if (m_filterLength)
			{
				if (!lengthFilter(contour, m_length_threshold)) {
					lengthFlag = false;
					continue;
				}
			}

			// contour area
			if (m_filterArea)
			{
				if (!areaFilter(contour, m_area_threshold)) {
					areaFlag = false;
					continue;
				}
			}

			// concatenate global filters outputs if several
			bool globalFilterOutput = true;

			if (m_filterNbOfPoints)
			{
				globalFilterOutput = nbOfPointsFlag;
			}

			if (m_filterLength)
			{
				globalFilterOutput = globalFilterOutput & lengthFlag;
			}

			if (m_filterArea)
			{
				globalFilterOutput = globalFilterOutput & areaFlag;
			}

			if (!globalFilterOutput)
			{
				continue;
			}
		}
		
		// pt filtering
		if (ptFiltering)
		{
			std::vector< cv::Point2f > pts;
			std::vector<float> direction;
			std::vector<float> response;
			cv::Vec4i hierarchy;
			std::vector<float> nx;
			std::vector<float> ny;
			double length;
			double area;

			// orientation filter data
			bool ptOrientationFlag=true;

			cv::Scalar color(rng.uniform(1, 254), rng.uniform(1, 254), rng.uniform(1, 254));
			cv::Vec4f line = contourOrientationLine(contour.points);
			cv::arrowedLine(output, cv::Point(int(line[2]), int(line[3])), cv::Point(int(line[2] + line[0] * 50), int(line[3] + line[1] * 50)), color);

			bool ptOtherFlag=true;

			for (int j = 0; j < (int)contour.points.size(); ++j)
			{
				cv::Point2f n = cv::Point2f(contour.nx[j], contour.ny[j]);

				// filter orientation
				if (m_filterOrientation)
				{
					cv::Point2f lineDir = cv::Point2f(line[2], line[3]);
					if (!orientationPtFilter(n, lineDir, m_orientationTolerance, m_angle_ref))
					{
						ptOrientationFlag = false;
					}
				}

				// concatenate point filters outputs if several
				bool ptFilterOutput = true;

				if (m_filterOrientation)
				{
					ptFilterOutput = ptOrientationFlag;
				}

				if (m_otherFilter)
				{
					ptFilterOutput = ptFilterOutput & ptOtherFlag;
				}

				if (ptFilterOutput)
				{
					pts.push_back(contour.points[j]);
					direction.push_back(contour.direction[j]);
					response.push_back(contour.response[j]);
					nx.push_back(contour.nx[j]);
					ny.push_back(contour.ny[j]);
				}
			}

			if (!pts.empty()) {
				hierarchy = contour.hierarchy;
				length = contour.length;
				area = contour.area;
				sp::EdgesSubPix::Contour filteredContour(pts, direction, response, hierarchy, nx, ny, length, area);
				validContours.push_back(filteredContour);
			}
		}

		cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
		cv::imshow(windowName, output);
	}

	if (!validContours.empty()) {
		contours = validContours;
	}
}

void sp::SubPix::updateEdgesListFromROIs()
{
	m_edgesInSubPixel.clear();
	for (auto it : m_roisEdgesPts) {
		m_edgesInSubPixel.insert(m_edgesInSubPixel.end(), it.second.begin(), it.second.end());
	}
}

void sp::SubPix::updateContoursListFromROIs()
{
	if (!m_roisContoursPtsFiltered.empty()) {
		m_roisContoursPts = m_roisContoursPtsFiltered;
	}

	m_contoursPtsInSubPixel.clear();
	for (auto it : m_roisContoursPts) {
		m_contoursPtsInSubPixel.insert(m_contoursPtsInSubPixel.end(), it.second.begin(), it.second.end());
	}
}

std::map< int, std::vector<sp::EdgesSubPix::Edge> > sp::SubPix::pixelEdgesMap(const std::vector<sp::EdgesSubPix::Edge>& edges, const std::vector<sp::EdgesSubPix::Edge>& subEdges, int imageHeight)
{
	std::cout << "pixelEdgesMap..." << std::endl;

    std::map< int, std::vector<sp::EdgesSubPix::Edge> > map;

	if (edges.size() != subEdges.size()) {
		return map;
	}

    for (int i = 0; i < (int)subEdges.size(); i++) {

        cv::Point2f pt = subEdges[i].point;
        float direction = subEdges[i].direction;
        float response = subEdges[i].response;

        int pt_x = int(pt.x);
        int pt_y = int(pt.y);

		// check if subpixel is on frontier
		if (pt_x == (edges[i].point.x + 1)) {
			pt_x--;
		}
		if (pt_y == (edges[i].point.y + 1)) {
			pt_y--;
		}

		int pt_id = pt_x*imageHeight + pt_y;

        // find pt
        std::map< int, std::vector<sp::EdgesSubPix::Edge> >::iterator mapIter = map.find(pt_id);
        if (mapIter == map.end()) {

            std::vector<sp::EdgesSubPix::Edge> subPts;
            sp::EdgesSubPix::Edge edge;
            edge.direction = direction;
            edge.point = pt;
            edge.response = response;
            subPts.push_back(edge);

            // add edge to map
            map[pt_id] = subPts;
        }
        else {

            // find edges
            sp::EdgesSubPix::Edge edge;
            edge.direction = direction;
            edge.point = pt;
            edge.response = response;
            mapIter->second.push_back(edge);
        }
    }

    return map;
}

std::map< int , std::map<int, std::vector<sp::EdgesSubPix::Edge> > > sp::SubPix::pixelContoursMap(const std::vector<sp::EdgesSubPix::Contour>& contours, const std::vector<sp::EdgesSubPix::Contour>& subContours, int imageHeight)
{
	std::cout << "pixelContoursMap..." << std::endl;

	std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > > map;

	if (contours.size() != subContours.size()) {
		return map;
	}

    for (int i = 0; i < (int)subContours.size(); i++) {

        std::vector< cv::Point2f > pts = subContours[i].points;
        std::vector< float > direction = subContours[i].direction;
        std::vector< float > response = subContours[i].response;
        for (int j = 0; j < (int)pts.size(); j++) {

			cv::Point2f pt = pts[j];
            int pt_x = int(pt.x);
            int pt_y = int(pt.y);

			// check if subpixel is on frontier
			if (pt_x == (contours[i].points[j].x + 1)) {
				pt_x--;
			}
			if (pt_y == (contours[i].points[j].y + 1)) {
				pt_y--;
			}

            int pt_id = pt_x*imageHeight + pt_y;

            // find pt
            std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > >::iterator mapIter = map.find(pt_id);
            if (mapIter == map.end()) {

                // create map for contours
                std::map<int, std::vector<sp::EdgesSubPix::Edge> > contourMap;

                // add pt to contour
                std::vector<sp::EdgesSubPix::Edge> subPts;
                sp::EdgesSubPix::Edge edge;
                edge.direction = direction[j];
                edge.point = pt;
                edge.response = response[j];
                subPts.push_back(edge);
                contourMap[i] = subPts;

                // add contour to map
                map[pt_id] = contourMap;
            }
            else {

                // find contour
                std::map<int, std::vector<sp::EdgesSubPix::Edge> >::iterator contourMapIter = mapIter->second.find(i);

                if (contourMapIter == mapIter->second.end())
                {
                    // add point from new contour
                    std::vector<sp::EdgesSubPix::Edge> subPts;
                    sp::EdgesSubPix::Edge edge;
                    edge.direction = direction[j];
                    edge.point = pt;
                    edge.response = response[j];
                    subPts.push_back(edge);
                    mapIter->second[i] = subPts;
                }
                else
                {
                    // add point from same contour
                    sp::EdgesSubPix::Edge edge;
                    edge.direction = direction[j];
                    edge.point = pt;
                    edge.response = response[j];
                    contourMapIter->second.push_back(edge);
                }
            }

        }

    }

    return map;
}

cv::Mat sp::SubPix::displayPixelState(const std::map<int, std::map<int, bool> >& pixelState, int imageWidth, int imageHeight, const std::string& windowName)
{
	std::cout << "displayPixelState..." << std::endl;

    cv::Mat frame(imageHeight, imageWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    for (std::map<int, std::map<int, bool> >::const_iterator pixelStateIter = pixelState.begin(); pixelStateIter != pixelState.end(); ++pixelStateIter) {

        int x = int(pixelStateIter->first / imageHeight);
        int y = int(pixelStateIter->first % imageHeight);

        // single subpixel, single contour by default
        cv::Mat_<cv::Vec3b> _frame = frame;
        _frame(y, x)[0] = 255;
        _frame(y, x)[1] = 255;
        _frame(y, x)[2] = 255;
        frame = _frame;

        // find pixel state
        std::map<int, bool>::const_iterator stateIter;

        bool singleSubpixelMultiContour = false;
        bool multiSubpixelSingleContour = false;
        bool multiSubpixelMultiContour = false;

        stateIter = pixelStateIter->second.find(PixelState::MULTISUBEDGES);
        if (stateIter != pixelStateIter->second.end())
        {
            if (!stateIter->second) {

                // single subpixel
                stateIter = pixelStateIter->second.find(PixelState::MULTICONTOUR);
                if (stateIter != pixelStateIter->second.end())
                {
                    // multi contours
                    if (stateIter->second) {
                        singleSubpixelMultiContour = true;
                    }
                }
            }
            else
            {
                // multi subpixel
                stateIter = pixelStateIter->second.find(PixelState::MULTICONTOUR);
                if (stateIter != pixelStateIter->second.end())
                {
                    if (!stateIter->second) {

                        // single contour
                        multiSubpixelSingleContour = true;
                    }
                    else
                    {
                        // multi contours
                        multiSubpixelMultiContour = true;
                    }
                }
                else
                {
                    // single contour by default
                    multiSubpixelSingleContour = true;
                }
            }
        }

        // affect color to each state
        if (singleSubpixelMultiContour) {
            cv::Mat_<cv::Vec3b> _frame = frame;
            _frame(y, x)[0] = 255;
            _frame(y, x)[1] = 0;
            _frame(y, x)[2] = 0;
            frame = _frame;

        }
        if (multiSubpixelSingleContour)
        {
            cv::Mat_<cv::Vec3b> _frame = frame;
            _frame(y, x)[0] = 0;
            _frame(y, x)[1] = 255;
            _frame(y, x)[2] = 0;
            frame = _frame;
        }
        if (multiSubpixelMultiContour)
        {
            cv::Mat_<cv::Vec3b> _frame = frame;
            _frame(y, x)[0] = 0;
            _frame(y, x)[1] = 0;
            _frame(y, x)[2] = 255;
            frame = _frame;
        }
    }

	// legend
	cv::String str_singleSubpixelSingleContour = "white: single subpixel, single contour";
	cv::String str_singleSubpixelMultiContour = "blue: single subpixel, multi contours";
	cv::String str_multiSubpixelSingleContour = "green: multi subpixels, single contour";
	cv::String str_multiSubpixelMultiContour = "red: multi subpixels, muli contours";
	int space = 20;
	cv::putText(frame, str_singleSubpixelSingleContour, cv::Point(space, space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1);
	cv::putText(frame, str_singleSubpixelMultiContour, cv::Point(space, 2 * space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1);
	cv::putText(frame, str_multiSubpixelSingleContour, cv::Point(space, 3 * space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);
	cv::putText(frame, str_multiSubpixelMultiContour, cv::Point(space, 4 * space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);

    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, frame);
    std::cout << "Displaying pixels state; press any key (in " << windowName << ") to continue.\n";

    return frame;
}

void sp::SubPix::computePixelState(std::map< int, std::vector<sp::EdgesSubPix::Edge> >& pixelEdgesMap, std::map<int, std::map<int, bool> >& pixelState)
{
	std::cout << "computePixelState..." << std::endl;

    std::map<int, bool> multiSubEdgesPixel;

    for (std::map< int, std::vector<sp::EdgesSubPix::Edge> > ::iterator iterMap = pixelEdgesMap.begin(); iterMap != pixelEdgesMap.end(); ++iterMap)
    {
        int pixel = iterMap->first;

        // check if pixel has several subpixel extractions
        std::vector<sp::EdgesSubPix::Edge> pts = iterMap->second;
		
		// remove duplicates
		auto comp = [](sp::EdgesSubPix::Edge& edge1, sp::EdgesSubPix::Edge& edge2) {  return cv::norm(edge1.point - edge2.point) < 1e-3f;  };
		auto uniquePts = std::unique(pts.begin(), pts.end(), comp);
		pts.erase(uniquePts, pts.end());

        if (pts.size() > 1)
        {
            multiSubEdgesPixel[pixel] = true;
        }
        else
        {
            multiSubEdgesPixel[pixel] = false;
        }
		iterMap->second = pts;

        std::map<int, bool> state;
        state[PixelState::MULTISUBEDGES] = multiSubEdgesPixel.find(pixel)->second;
        pixelState[pixel] = state;
    }
}

void sp::SubPix::computePixelState(std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > >& pixelContoursMap, std::map<int, std::map<int, bool> >& pixelState)
{
	std::cout << "computePixelState..." << std::endl;

    std::map<int, bool> multicontoursPixel;
    std::map<int, bool> multiSubEdgesPixel;
    for (std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > >::const_iterator iterMap = pixelContoursMap.begin(); iterMap != pixelContoursMap.end(); ++iterMap)
    {
        int pixel = iterMap->first;
        multiSubEdgesPixel[pixel] = false;
    }

    for (std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > >::iterator iterMap = pixelContoursMap.begin(); iterMap != pixelContoursMap.end(); ++iterMap)
    {
        int pixel = iterMap->first;

        // check if pixel is on several contours
        if (iterMap->second.size()>1)
        {
            multicontoursPixel[pixel] = true;
        }
        else
        {
            multicontoursPixel[pixel] = false;
        }

        for (std::map<int, std::vector<sp::EdgesSubPix::Edge> >::iterator iterContour = iterMap->second.begin(); iterContour != iterMap->second.end(); ++iterContour)
        {
            // create a copy not to modify pixel map
            std::vector<sp::EdgesSubPix::Edge> pts = iterContour->second;

            // remove duplicates
            auto comp = [](sp::EdgesSubPix::Edge& edge1, sp::EdgesSubPix::Edge& edge2) {  return cv::norm(edge1.point - edge2.point) < 1e-3f;  };
            auto uniquePts = std::unique(pts.begin(), pts.end(), comp);
            pts.erase(uniquePts, pts.end());
            if (pts.size() > 1)
            {
                multiSubEdgesPixel[pixel] = true;
            }
			iterContour->second = pts;
        }
        std::map<int, bool> state;
        state[PixelState::MULTICONTOUR] = multicontoursPixel.find(pixel)->second;
        state[PixelState::MULTISUBEDGES] = multiSubEdgesPixel.find(pixel)->second;
        pixelState[pixel] = state;
    }
}

void sp::SubPix::contourVec2VecForDisplay(const std::vector<sp::EdgesSubPix::Contour>& contoursPts, std::vector < std::vector< cv::Point > >& contours, std::vector < cv::Vec4i >& hierarchy)
{
    contours.clear();
	hierarchy.clear();
    for (int i = 0; i < (int)contoursPts.size(); ++i)
    {
        std::vector< cv::Point2f> pts = contoursPts[i].points;
        std::vector < cv::Point > ptsInt;
        for (int j = 0; j < (int)pts.size(); j++) {
            cv::Point2f pt(pts[j].x, pts[j].y);
			cv::Point pt2Display((int)pt.x, (int)pt.y);
            ptsInt.push_back(pt2Display);
        }
        contours.push_back(ptsInt);
		hierarchy.push_back(contoursPts[i].hierarchy);
    }
}

void sp::SubPix::show(cv::Mat& image, const std::string& windowName)
{
	if (!image.empty()) {

		cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
		cv::imshow(windowName, image);
	}
}

void sp::SubPix::drawContour(const int& contourId, const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& contoursImage, const cv::Scalar& color, bool markers, int markerType, bool normals, int markersDisplayRatio, int normalsDisplayRatio, int contourThickness)
{
    if (contourId < (int)contoursPts.size()) {

        if (markers) {
            for (int j = 0; j < contoursPts[contourId].points.size(); j += markersDisplayRatio) {
				cv::Point2f pt(contoursPts[contourId].points[j].x, contoursPts[contourId].points[j].y);
                cv::drawMarker(contoursImage, pt, color, markerType, contourThickness, 1);
				if (normals) {
					if (j % normalsDisplayRatio == 0) {
						cv::Point2f dst = contoursPts[contourId].points[j] + cv::Point2f(10 * contoursPts[contourId].nx[j], 10 * contoursPts[contourId].ny[j]);
						cv::arrowedLine(contoursImage, contoursPts[contourId].points[j], dst, color, 1, 8);
					}
				}
            }
        }
        else {

            std::vector < std::vector< cv::Point > > contours;
			std::vector < cv::Vec4i > hierarchy;
            contourVec2VecForDisplay(contoursPts, contours, hierarchy);
            cv::drawContours(contoursImage, contours, contourId, color, 1, 8);

			for (int j = 0; j < contoursPts[contourId].points.size(); j+= normalsDisplayRatio) {
			
				if (normals) {
					cv::Point2f dst = contoursPts[contourId].points[j] + cv::Point2f(10 * contoursPts[contourId].nx[j], 10 * contoursPts[contourId].ny[j]);
					cv::arrowedLine(contoursImage, contoursPts[contourId].points[j], dst, color, 1, 8);
				}
			}
        }
    }
}

void sp::SubPix::drawContours(const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& contoursImage)
{
    cv::RNG rng;

    std::vector < std::vector< cv::Point > > contours;
	std::vector < cv::Vec4i > hierarchy;
	contourVec2VecForDisplay(contoursPts, contours, hierarchy);

    for (int i = 0; i < (int)contours.size(); i++) {
        cv::Scalar color(rng.uniform(1, 254), rng.uniform(1, 254), rng.uniform(1, 254));
        cv::drawContours(contoursImage, contours, i, color, 1, 8);
    }
}

void sp::SubPix::showContours(const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& contoursImage, const std::string& windowName, bool markers, int markerType, bool normals, int markersDisplayRatio, int normalsDisplayRatio, int contourThickness)
{
    if (!contoursImage.empty()) {

        if (markers) {

            cv::RNG rng;

            for (int i = 0; i < (int)contoursPts.size(); ++i)
            {
                //if (i == 50)
                {
                    // set color
                    cv::Scalar color(rng.uniform(1, 254), rng.uniform(1, 254), rng.uniform(1, 254));

                    // draw contour
                    drawContour(i, contoursPts, contoursImage, color, markers, markerType, normals, markersDisplayRatio, normalsDisplayRatio, contourThickness);
                }
            }
        }
        else {
            drawContours(contoursPts, contoursImage);
        }

        // show contours
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        cv::imshow(windowName, contoursImage);
    }
}

void sp::SubPix::showContour(const int& contourId, const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& contoursImage, const std::string& windowName, const cv::Scalar& color, bool marker, int markerType, bool normals, int markersDisplayRatio, int normalsDisplayRatio, bool selectContourStepByStep, int contourThickness)
{
	if (selectContourStepByStep)
	{
		// draw contour
		drawContour(contourId, contoursPts, contoursImage, color, false, cv::MARKER_CROSS, false, 10, 10, contourThickness);

		// show contours
		cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
		cv::imshow(windowName, contoursImage);

		if (m_display) {
			std::cout << "Displaying the contour's pts; press any key (in " << windowName << ") to continue." << std::endl;
		}
		
		sp::EdgesSubPix::Contour contour = contoursPts[contourId];
		if (m_contourStepId != contourId)
		{
			m_contourStep = 0;
		}
		m_contourStepId = contourId;

		cv::Point2f pt(contour.points[m_contourStep].x, contour.points[m_contourStep].y);
		cv::Point2f n(contour.nx[m_contourStep], contour.ny[m_contourStep]);
		cv::drawMarker(contoursImage, pt, color, markerType, 10);
		if (normals) {
			cv::Point2f dst = pt + n;
			cv::arrowedLine(contoursImage, pt, dst, color, 1, 8);
		}

		// show contours step by step
		cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
		cv::imshow(windowName, contoursImage);

		// clean contours view
		cv::cvtColor(m_image, contoursImage, cv::COLOR_GRAY2BGR);
		
		m_contourStep++;

		if (m_contourStep == contour.points.size()) {
			m_selectContourStepByStep = false;
		}
	}
	else
	{
		// reset contour step counter
		m_contourStep = 0;

		// draw contour
		drawContour(contourId, contoursPts, contoursImage, color, marker, markerType, normals, markersDisplayRatio, normalsDisplayRatio, contourThickness);

		// show contours
		cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
		cv::imshow(windowName, contoursImage);
	}
}

void sp::SubPix::parseInputPoints2YmlFile(const std::string& filename, std::vector<cv::Point>& inputPoints)
{
    if (!filename.empty() && !inputPoints.empty()) {
        cv::FileStorage fs(filename, cv::FileStorage::READ);

        cv::FileNode points = fs["points"];
        cv::FileNodeIterator it = points.begin(), it_end = points.end();
        int idx = 0;

        // iterate through a sequence using FileNodeIterator
        for (; it != it_end; ++it, idx++)
        {
            inputPoints.push_back(cv::Point((int)(*it)["x"], (int)(*it)["y"]));
        }
        fs.release();
    }
}

void sp::SubPix::saveEdges2YmlFile(const std::string& filename, cv::Mat& edges)
{
    if (!filename.empty() && !edges.empty()) {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
        fs << "edges" << "[";
        for (int i = 0; i < edges.rows; ++i) {
            for (int j = 0; j < edges.cols; ++j) {
                if (edges.at<uchar>(i, j)) {
                    fs << "{:";
                    fs << "id" << j*edges.rows+i << "x" << j << "y" << i;
                    fs << "}";
                }
            }
        }
        fs << "]";
        fs.release();
    }
}

void sp::SubPix::saveEdges2YmlFile(const std::string& filename, const std::vector<sp::EdgesSubPix::Edge>& edges)
{
    if (!filename.empty() && !edges.empty())
    {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
        fs << "edges" << "[";
        for (size_t i = 0; i < edges.size(); ++i)
        {
            fs << "{:";
            fs << "id" << (int)i;
            fs << "point" << edges[i].point << "response" << edges[i].response << "direction" << edges[i].direction;
            fs << "}";
        }
        fs << "]";
        fs.release();
    }
}

void sp::SubPix::saveImage(const std::string& filename, cv::Mat& image)
{
    if (!filename.empty() & !image.empty()) {
        std::vector<int> compression;
        compression.push_back(CV_IMWRITE_PXM_BINARY);
        compression.push_back(0);
        cv::imwrite(filename, image, compression);
    }
}

void sp::SubPix::saveContours2YmlFile(const std::string& filename, const std::vector<sp::EdgesSubPix::Contour>& contours)
{
    if (!filename.empty() && !contours.empty())
    {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
        fs << "contours" << "[";
        for (size_t i = 0; i < contours.size(); ++i)
        {
            fs << "{:";
            fs << "id" << (int)i;
            fs << "points" << contours[i].points;
            fs << "points size" << (int)contours[i].points.size();
            fs << "response" << contours[i].response;
            fs << "response size" << (int)contours[i].response.size();
            fs << "direction" << contours[i].direction;
            fs << "direction size" << (int)contours[i].direction.size();
            fs << "}";
        }
        fs << "]";
        fs.release();
    }
}

void sp::SubPix::setROIs(const std::vector<cv::Point>& inputPoints, const int& defaultSize)
{
    if (!inputPoints.empty()) {
        m_rois.clear();
    }

    for (int i = 0; (int)i < inputPoints.size(); i++) {

        // ROI name
        std::stringstream roiName;
        roiName << i;

        cv::Size size(defaultSize, defaultSize);
        m_rois[roiName.str()] = cv::Rect(inputPoints[i].x - size.width / 2, inputPoints[i].y - size.height / 2, size.width, size.height);
    }
}

void sp::SubPix::setROIs(const std::map< std::string, cv::Rect>& rois)
{
    m_rois = rois;
}

void sp::SubPix::setGrayImage(const cv::Mat& grayImage)
{
    m_image = grayImage;
}

void sp::SubPix::setEdgesImage(const cv::Mat& edges)
{ 
    m_edges = edges;
}

void sp::SubPix::setEdgesMask(const cv::Mat & mask)
{
	m_mask = mask;
}

void sp::SubPix::setROI(const std::string& roiName, const cv::Rect& roi)
{
    m_rois[roiName] = roi;
}

const cv::Rect sp::SubPix::getROI(const std::string& roiName)
{
    std::map< std::string, cv::Rect>::const_iterator roi = m_rois.find(roiName);
    if (roi != m_rois.end())
    {
        return roi->second;
    }
    else {
        return cv::Rect();
    }
}

void sp::SubPix::setImageROI(const std::string& roiName, const cv::Rect& roi)
{
    cv::Mat imageROI = m_image(roi);
    cv::Mat edgesROI = m_edges(roi);

    std::map < std::string, cv::Mat > image_rois;
    image_rois[INPUT_GRAYIMAGE] = imageROI;
    image_rois[EDGES_IMAGE] = edgesROI;

    m_roiImages[roiName] = image_rois;
}

void sp::SubPix::getImageROI(const std::string& roiName, cv::Mat& imageROI, cv::Mat& edgesROI)
{
    // get rois
    auto rois = m_roiImages.find(roiName);
    if (rois != m_roiImages.end()) {
        auto imageROIIter = rois->second.find(INPUT_GRAYIMAGE);
        auto edgesROIIter = rois->second.find(EDGES_IMAGE);
        if ((imageROIIter != rois->second.end()) && (edgesROIIter != rois->second.end())) {
            imageROI = imageROIIter->second;
            edgesROI = edgesROIIter->second;
        }
    }
}

void sp::SubPix::setImageROIs(const std::map< std::string, cv::Rect >& rois, const cv::Mat& grayImage)
{
    if (!grayImage.empty())
    {
        setGrayImage(grayImage);
        m_edges = cv::Mat::zeros(m_image.size(), CV_8UC1);
    }

    if (!m_image.empty()) {

        if (m_edges.empty())
        {
            m_edges = cv::Mat::zeros(m_image.size(), CV_8UC1);
        }

        m_rois = rois;

        m_roiImages.clear();

        for (auto i : m_rois)
        {
            // roiName
            std::stringstream roiName;
            roiName << i.first;

            // roiGeometry
            cv::Rect roi = i.second;

            setImageROI(roiName.str(), roi);
        }

        if (m_roiImages.empty()) {
            setROI("Whole Image", cv::Rect(0, 0, m_image.cols, m_image.rows));
            setImageROI("Whole Image", cv::Rect(0, 0, m_image.cols, m_image.rows));
        }
    }
}

void sp::SubPix::destroyWindows()
{
    cv::destroyWindow(m_EDGES_WINDOW_NAME);
	cv::destroyWindow(m_EDGES_FROM_CONTOURS);
    cv::destroyWindow(m_PIXEL_STATE_AFTER_CONTOURS_DETECTION_WINDOW_NAME);
    cv::destroyWindow(m_CONTOURS_WINDOW_NAME);
	cv::destroyWindow(m_FILTERED_CONTOURS_WINDOW_NAME);
    cv::destroyWindow(m_PIXEL_STATE_AFTER_EDGES_DETECTION_WINDOW_NAME);
    cv::destroyWindow(m_MOVING_EDGES_WINDOW_NAME);
    cv::destroyWindow(m_AMBIGUITY_ON_EDGES_IMAGE_LIST_WINDOW_NAME);
	cv::destroyWindow(m_AMBIGUITY_ON_CONTOURS_IMAGE_LIST_WINDOW_NAME);
}

void sp::SubPix::saveEdgesPixelMap(const std::string& filename, int imageHeight, std::map< int, std::vector<sp::EdgesSubPix::Edge > >& pixelEdgesMap)
{
    if (!filename.empty() && !pixelEdgesMap.empty())
    {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
        fs << "edges pixel map" << "[";
        for (std::map< int, std::vector<sp::EdgesSubPix::Edge > >::iterator pixelEdgesMapIter = pixelEdgesMap.begin(); pixelEdgesMapIter != pixelEdgesMap.end(); ++pixelEdgesMapIter)
        {
            fs << "{:";
            fs << "x" << int(pixelEdgesMapIter->first / imageHeight) << "y" << int(pixelEdgesMapIter->first % imageHeight);
            fs << "}";
            for (std::vector<sp::EdgesSubPix::Edge >::iterator edgesIter = pixelEdgesMapIter->second.begin(); edgesIter != pixelEdgesMapIter->second.end(); ++edgesIter)
            {
                fs << "{:";
                fs << "points" << (*edgesIter).point;
                fs << "response" << (*edgesIter).response;
                fs << "direction" << (*edgesIter).direction;
                fs << "}";
            }
        }
        fs << "]";
        fs.release();
    }
}

void sp::SubPix::saveContoursPixelMap(const std::string& filename, int imageHeight, std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > >& pixelContoursMap)
{
    if (!filename.empty() && !pixelContoursMap.empty())
    {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
        fs << "contours pixel map" << "[";
        for (std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > >::const_iterator pixelContoursMapIter = pixelContoursMap.begin(); pixelContoursMapIter != pixelContoursMap.end(); ++pixelContoursMapIter)
        {
            fs << "{:";
            fs << "x" << int(pixelContoursMapIter->first / imageHeight) << "y" << int(pixelContoursMapIter->first % imageHeight);
            fs << "}";

            for (std::map<int, std::vector<sp::EdgesSubPix::Edge> >::const_iterator contoursMapIter = pixelContoursMapIter->second.begin(); contoursMapIter != pixelContoursMapIter->second.end(); ++contoursMapIter)
            {
                std::stringstream ss;
                ss << contoursMapIter->first;
                fs << "contour " + ss.str() << "[";

                for (std::vector<sp::EdgesSubPix::Edge >::const_iterator edgesIter = contoursMapIter->second.begin(); edgesIter != contoursMapIter->second.end(); ++edgesIter) {
                    fs << "{:";
                    fs << "points" << (*edgesIter).point;
                    fs << "response" << (*edgesIter).response;
                    fs << "direction" << (*edgesIter).direction;
                    fs << "}";
                }

                fs << "]";
            }
        }
        fs << "]";
        fs.release();
    }
}

cv::Mat sp::SubPix::displayMovingEdges(const std::vector<sp::EdgesSubPix::Edge>& edgesInPixel, const std::vector<sp::EdgesSubPix::Edge>& edgesInSubPixel, int imageWidth, int imageHeight, const std::string & windowName)
{
	std::cout << "displayMovingEdges..." << std::endl;

    // extractions must correspond
    if (edgesInPixel.size() != edgesInSubPixel.size()) {
        return cv::Mat();
    }

    cv::Mat movingEdges(imageHeight, imageWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i=0; i<(int)edgesInPixel.size(); i++)
    {
		cv::Point2f pt = edgesInSubPixel[i].point;
        int x = int(pt.x);
        int y = int(pt.y);
        float res_x = cv::abs(x - pt.x);
        float res_y = cv::abs(y - pt.y);

        cv::Mat_<cv::Vec3b> _frame = movingEdges;
        if ((x == edgesInPixel[i].point.x) && (y == edgesInPixel[i].point.y))
        {
            if (res_x < 1e-3 && res_y < 1e-3)
            {
                // no subpixel
                _frame(y, x)[0] = 0;
                _frame(y, x)[1] = 0;
                _frame(y, x)[2] = 255;
            }
            else {
                // subpixel is within pixel
                _frame(y, x)[0] = 0;
                _frame(y, x)[1] = 255;
                _frame(y, x)[2] = 0;
            }

        }
        else
        {
            // subpixel is either at pixel right edge or down edge (if edge is exactly between pixels)
            if (  ( (edgesInPixel[i].point.x+1) == x ) || ( (edgesInPixel[i].point.y+1) == y ) ) {

                // subpixel is at next pixel frontier
                _frame(y, x)[0] = 255;
                _frame(y, x)[1] = 255;
                _frame(y, x)[2] = 0;
            }
            else {
                // subpixel is totally false (not occurring normally)
                _frame(y, x)[0] = 255;
                _frame(y, x)[1] = 0;
                _frame(y, x)[2] = 0;
            }
        }
        movingEdges = _frame;
    }

	// legend
	cv::String noSubpixel = "red: no subpixel";
	cv::String withinPixel = "green: within pixel";
	cv::String atPixelFrontier = "cyan: at pixel frontier";
	cv::String outOfPixel = "blue: out of pixel";
	int space = 20;
	cv::putText(movingEdges, noSubpixel, cv::Point(space, space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);
	cv::putText(movingEdges, withinPixel, cv::Point(space, 2*space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);
	cv::putText(movingEdges, atPixelFrontier, cv::Point(space, 3*space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0), 1);
	cv::putText(movingEdges, outOfPixel, cv::Point(space, 4*space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1);

    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, movingEdges);
    std::cout << "Displaying moving edges; press any key (in " << windowName << ") to continue.\n";

    return movingEdges;
}

cv::Mat sp::SubPix::displayMovingContourEdges(const std::vector<sp::EdgesSubPix::Contour>& contoursInPixel, const std::vector<sp::EdgesSubPix::Contour>& contoursInSubPixel, int imageWidth, int imageHeight, const std::string & windowName)
{
	std::cout << "displayMovingEdges..." << std::endl;

	// extractions must correspond
	if (contoursInPixel.size() != contoursInSubPixel.size()) {
		return cv::Mat();
	}

	cv::Mat movingEdges(imageHeight, imageWidth, CV_8UC3, cv::Scalar(0, 0, 0));

	for (int i = 0; i<(int)contoursInPixel.size(); i++)
	{
		for (int j = 0; j < (int)contoursInPixel[i].points.size(); j++)
		{
			cv::Point2f pt = contoursInSubPixel[i].points[j];
			int x = int(pt.x);
			int y = int(pt.y);
			float res_x = cv::abs(x - pt.x);
			float res_y = cv::abs(y - pt.y);

			cv::Mat_<cv::Vec3b> _frame = movingEdges;
			if ((x == contoursInPixel[i].points[j].x) && (y == contoursInPixel[i].points[j].y))
			{
				if (res_x < 1e-3 && res_y < 1e-3)
				{
					// no subpixel
					_frame(y, x)[0] = 0;
					_frame(y, x)[1] = 0;
					_frame(y, x)[2] = 255;
				}
				else {
					// subpixel is within pixel
					_frame(y, x)[0] = 0;
					_frame(y, x)[1] = 255;
					_frame(y, x)[2] = 0;
				}

			}
			else
			{
				// subpixel is either at pixel right edge or down edge (if edge is exactly between pixels)
				if (((contoursInPixel[i].points[j].x + 1) == x) || ((contoursInPixel[i].points[j].y + 1) == y)) {

					// subpixel is at next pixel frontier
					_frame(y, x)[0] = 255;
					_frame(y, x)[1] = 255;
					_frame(y, x)[2] = 0;
				}
				else {
					// subpixel is totally false (not occurring normally)
					_frame(y, x)[0] = 255;
					_frame(y, x)[1] = 0;
					_frame(y, x)[2] = 0;
				}
			}
			movingEdges = _frame;
		}
	}

	// legend
	cv::String noSubpixel = "red: no subpixel";
	cv::String withinPixel = "green: within pixel";
	cv::String atPixelFrontier = "cyan: at pixel frontier";
	cv::String outOfPixel = "blue: out of pixel";
	int space = 20;
	cv::putText(movingEdges, noSubpixel, cv::Point(space, space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);
	cv::putText(movingEdges, withinPixel, cv::Point(space, 2 * space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);
	cv::putText(movingEdges, atPixelFrontier, cv::Point(space, 3 * space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0), 1);
	cv::putText(movingEdges, outOfPixel, cv::Point(space, 4 * space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1);

    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
	cv::imshow(windowName, movingEdges);
	std::cout << "Displaying moving edges; press any key (in " << windowName << ") to continue.\n";

	return movingEdges;
}

cv::Mat sp::SubPix::displayImageSequenceEdgesAmbiguities(int imageWidth, int imageHeight, const std::string & windowName)
{
	std::cout << "displayImageSequenceEdgesAmbiguities..." << std::endl;

	cv::Mat ambiguityImage(imageHeight, imageWidth, CV_8UC3, cv::Scalar(0, 0, 0));
	std::map < int, std::vector<sp::EdgesSubPix::Edge> > ambiguitySubEdges;

	for (std::map < cv::String, std::map< int, std::vector<sp::EdgesSubPix::Edge> > >::iterator imageListEdgesAmbiguitiesIter = m_imageListEdgesAmbiguities.begin(); imageListEdgesAmbiguitiesIter != m_imageListEdgesAmbiguities.end(); ++imageListEdgesAmbiguitiesIter)
	{
		for (std::map< int, std::vector<sp::EdgesSubPix::Edge> >::iterator edgesIter = imageListEdgesAmbiguitiesIter->second.begin(); edgesIter != imageListEdgesAmbiguitiesIter->second.end(); ++edgesIter) {

			for (std::vector<sp::EdgesSubPix::Edge>::iterator iterPts = edgesIter->second.begin(); iterPts != edgesIter->second.end(); ++iterPts) {
				ambiguitySubEdges[edgesIter->first].push_back(*iterPts);
			}
		}
	}

	for (std::map < int, std::vector<sp::EdgesSubPix::Edge> >::iterator ambiguitiesIter = ambiguitySubEdges.begin(); ambiguitiesIter != ambiguitySubEdges.end(); ++ambiguitiesIter)
	{
		int x = int(ambiguitiesIter->first / imageHeight);
		int y = int(ambiguitiesIter->first % imageHeight);

		cv::Mat_<cv::Vec3b> _frame = ambiguityImage;

		std::vector< sp::EdgesSubPix::Edge> pts= ambiguitiesIter->second;
		auto comp = [](sp::EdgesSubPix::Edge& edge1, sp::EdgesSubPix::Edge& edge2) {  return cv::norm(edge1.point - edge2.point) < 1e-3f;  };
		auto uniquePts = std::unique(pts.begin(), pts.end(), comp);
		pts.erase(uniquePts, pts.end());

		if (pts.size() > 1)
		{
			// pixel has multi subedges
			_frame(y, x)[0] = 0;
			_frame(y, x)[1] = 0;
			_frame(y, x)[2] = 255;
		}
		else {
			// pixel has only one subedge
			_frame(y, x)[0] = 255;
			_frame(y, x)[1] = 255;
			_frame(y, x)[2] = 255;
		}
		ambiguityImage = _frame;
	}

	// legend
	cv::String ambiguity = "red: ambiguities on image sequence edges";
	cv::String noAmbiguity = "white: no ambiguity on image sequence edges";
	int space = 20;
	cv::putText(ambiguityImage, ambiguity, cv::Point(space, space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);
	cv::putText(ambiguityImage, noAmbiguity, cv::Point(space, 2 * space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1);

	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
	cv::imshow(windowName, ambiguityImage);
	std::cout << "Displaying ambiguity images; press any key (in " << windowName << ") to continue.\n";

	return ambiguityImage;
}

cv::Mat sp::SubPix::displayImageSequenceContoursAmbiguities(int imageWidth, int imageHeight, const std::string & windowName)
{
	std::cout << "displayImageSequenceContoursAmbiguities..." << std::endl;

	cv::Mat ambiguityImage(imageHeight, imageWidth, CV_8UC3, cv::Scalar(0, 0, 0));
	std::map < int, std::vector<sp::EdgesSubPix::Edge> > ambiguitySubEdges;

	for (std::map < cv::String, std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > > >::iterator imageListContoursAmbiguitiesIter = m_imageListContoursAmbiguities.begin(); imageListContoursAmbiguitiesIter != m_imageListContoursAmbiguities.end(); ++imageListContoursAmbiguitiesIter)
	{
		for (std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > >::iterator edgesIter = imageListContoursAmbiguitiesIter->second.begin(); edgesIter != imageListContoursAmbiguitiesIter->second.end(); ++edgesIter) {

			// extract edges from multi contours
			std::vector< sp::EdgesSubPix::Edge> pts;
			for (std::map<int, std::vector<sp::EdgesSubPix::Edge> >::iterator contoursIter = edgesIter->second.begin(); contoursIter != edgesIter->second.end(); ++contoursIter) {
				for (std::vector<sp::EdgesSubPix::Edge>::const_iterator contoursPtsIter = contoursIter->second.begin(); contoursPtsIter != contoursIter->second.end(); ++contoursPtsIter) {
					ambiguitySubEdges[edgesIter->first].push_back(*contoursPtsIter);
				}
			}
		}
	}

	for (std::map < int, std::vector<sp::EdgesSubPix::Edge> >::iterator ambiguitiesIter = ambiguitySubEdges.begin(); ambiguitiesIter != ambiguitySubEdges.end(); ++ambiguitiesIter)
	{
		int x = int(ambiguitiesIter->first / imageHeight);
		int y = int(ambiguitiesIter->first % imageHeight);

		cv::Mat_<cv::Vec3b> _frame = ambiguityImage;

		// extract edges from multi contours
		std::vector< sp::EdgesSubPix::Edge> pts = ambiguitiesIter->second;
		auto comp = [](sp::EdgesSubPix::Edge& edge1, sp::EdgesSubPix::Edge& edge2) {  return cv::norm(edge1.point - edge2.point) < 1e-3f;  };
		auto uniquePts = std::unique(pts.begin(), pts.end(), comp);
		pts.erase(uniquePts, pts.end());

		if (pts.size() > 1)
		{
			// pixel has multi subedges
			_frame(y, x)[0] = 0;
			_frame(y, x)[1] = 0;
			_frame(y, x)[2] = 255;
		}
		else {
			// pixel has only one subedge
			_frame(y, x)[0] = 255;
			_frame(y, x)[1] = 255;
			_frame(y, x)[2] = 255;
		}
		ambiguityImage = _frame;
	}

	// legend
	cv::String ambiguity = "red: ambiguities on image sequence contours edges";
	cv::String noAmbiguity = "white: no ambiguity on image sequence contours edges";
	int space = 20;
	cv::putText(ambiguityImage, ambiguity, cv::Point(space, space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);
	cv::putText(ambiguityImage, noAmbiguity, cv::Point(space, 2 * space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1);

	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
	cv::imshow(windowName, ambiguityImage);
	std::cout << "Displaying ambiguity images; press any key (in " << windowName << ") to continue.\n";

	return ambiguityImage;
}
