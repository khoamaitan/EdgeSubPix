#include <iterator>

#include "EdgesSubPix.h"
#include "SubPix.h"






sp::SubPix::SubPix()
{
}


sp::SubPix::~SubPix()
{
}

cv::Point2f sp::SubPix::subpixel2Image(const cv::Point2f& subpixel)
{
	cv::Point2f pt;
	pt.x = subpixel.x+0.5f;
	pt.y = subpixel.y+0.5f;

	return pt;
}

cv::Point2f sp::SubPix::image2Subpixel(const cv::Point2f& imageCoord)
{
	cv::Point2f pt;
	pt.x = imageCoord.x-0.5f;
	pt.y = imageCoord.y-0.5f;

	return pt;
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
        m_edgesSubPix.edgesSubPix(imageROI, m_alpha, m_low, m_high, m_edgesInPixel, m_edgesInSubPixel, edgesROI);
        int64 t1 = cv::getCPUTickCount();

        cv::Rect roi = getROI(roiImage.first);

        if (m_display) {
            if (!roi.empty()) {
                std::cout << "ROI " << roiImage.first << " (" << roi.x << "," << roi.y << "," << roi.width << "x" << roi.height << ")" << " : " << "execution time is " << (t1 - t0) / (double)cv::getTickFrequency() << " seconds" << std::endl;
                std::cout << "nb edges (subpixel) : " << m_edgesInSubPixel.size() << std::endl;
                std::cout << "nb edges (pixel) : " << m_edgesInPixel.size() << std::endl;
            }
        }

        m_roisEdgesPts[roiImage.first] = m_edgesInSubPixel;

        m_nbOfEdges += (int)m_edgesInSubPixel.size();
    }

    // show contours
    if (m_display) {

        // show edges
        showEdges(m_edges, m_EDGES_WINDOW_NAME);
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
        m_edgesSubPix.edgesSubPix(imageROI, m_alpha, m_low, m_high, m_edgesInPixel, m_contoursPtsInPixel, m_contoursPtsInSubPixel, m_hierarchy, m_contourMode, edgesROI);
        int64 t1 = cv::getCPUTickCount();

        if (m_display) {

            cv::Rect roi = getROI(roiImage.first);
            if (!roi.empty()) {
                std::cout << "ROI " << roiImage.first << " (" << roi.x << "," << roi.y << "," << roi.width << "x" << roi.height << ")" << " : " << "execution time is " << (t1 - t0) / (double)cv::getTickFrequency() << " seconds" << std::endl;
                std::cout << "nb contours (subpixel) : " << m_contoursPtsInSubPixel.size() << std::endl;
                std::cout << "nb contours (pixel) : " << m_contoursPtsInPixel.size() << std::endl;
            }
        }

        m_roisContoursPts[roiImage.first] = m_contoursPtsInSubPixel;
        m_roisHierarchy[roiImage.first] = m_hierarchy;

        m_nbOfContours += (int)m_contoursPtsInSubPixel.size();
    }

    // show contours
    if (m_display) {

        // show edges
        showEdges(m_edges, m_EDGES_WINDOW_NAME);

        // limitation due to openCV bad profiling
        if (m_nbOfContours < m_maxContours) {
            cv::cvtColor(m_image, m_contours, cv::COLOR_GRAY2RGB);

            for (auto roiImage : m_roiImages) {

                auto contoursPts = m_roisContoursPts.find(roiImage.first);
                auto hierarchy = m_roisHierarchy.find(roiImage.first);
                showContours(contoursPts->second, m_image, m_contours, m_CONTOURS_WINDOW_NAME, hierarchy->second);
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
    std::map< std::string, std::vector< sp::EdgesSubPix::Contour> >::const_iterator roiContoursPts;
    for (roiContoursPts = m_roisContoursPts.begin(); roiContoursPts != m_roisContoursPts.end(); ++roiContoursPts)
    {
        countoursNumber += (int)roiContoursPts->second.size();
        if (threshold < countoursNumber) {
            break;
        }
    }

    int contourId = (roiContoursPts == m_roisContoursPts.begin()) ? threshold : countoursNumber - threshold - 1;
    showContour(contourId, roiContoursPts->second, m_contours, m_CONTOURS_WINDOW_NAME, color);

    m_selectedContour = m_contoursPtsInSubPixel[contourId];
	m_selectedContourInPixel = m_contoursPtsInPixel[contourId];

    std::cout << "selected contour " << contourId << std::endl;
	std::cout << "nb of points pixel resolution : " << m_selectedContourInPixel.points.size() << std::endl;
    std::cout << "nb of points subpixel resolution : " << m_selectedContour.points.size() << std::endl;
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

    if (cv::getWindowProperty(windowName, cv::WindowPropertyFlags::WND_PROP_VISIBLE) == -1) {
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);
    }
    cv::imshow(windowName, frame);
    std::cout << "Displaying pixels state; press any key (in " << windowName << ") to continue.\n";

    return frame;
}

void sp::SubPix::computePixelState(std::map< int, std::vector<sp::EdgesSubPix::Edge> >& pixelEdgesMap, std::map<int, std::map<int, bool> >& pixelState)
{
	std::cout << "computePixelState..." << std::endl;

    std::map<int, bool> multiSubEdgesPixel;

    for (std::map< int, std::vector<sp::EdgesSubPix::Edge> > ::const_iterator iterMap = pixelEdgesMap.begin(); iterMap != pixelEdgesMap.end(); ++iterMap)
    {
        int pixel = iterMap->first;

        // check if pixel has several subpixel extractions
        std::vector<sp::EdgesSubPix::Edge> pts = iterMap->second;
        if (pts.size() > 1)
        {
            multiSubEdgesPixel[pixel] = true;
        }
        else
        {
            multiSubEdgesPixel[pixel] = false;
        }

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
            std::vector<sp::EdgesSubPix::Edge> pts;
            std::copy(iterContour->second.begin(), iterContour->second.end(), std::back_inserter(pts));

            // remove duplicates
            auto comp = [](sp::EdgesSubPix::Edge& edge1, sp::EdgesSubPix::Edge& edge2) {  return cv::norm(edge1.point - edge2.point) < 1e-3f;  };
            auto uniquePts = std::unique(pts.begin(), pts.end(), comp);
            pts.erase(uniquePts, pts.end());
            if (pts.size() > 1)
            {
                multiSubEdgesPixel[pixel] = true;
            }
        }
        std::map<int, bool> state;
        state[PixelState::MULTICONTOUR] = multicontoursPixel.find(pixel)->second;
        state[PixelState::MULTISUBEDGES] = multiSubEdgesPixel.find(pixel)->second;
        pixelState[pixel] = state;
    }
}

void sp::SubPix::contourVec2PtsVecVecForDisplay(const std::vector<sp::EdgesSubPix::Contour>& contoursPts, std::vector < std::vector< cv::Point > >& contours)
{
    contours.clear();
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
    }
}

void sp::SubPix::showEdges(cv::Mat& edges, const std::string& windowName)
{
    if (cv::getWindowProperty(windowName, cv::WindowPropertyFlags::WND_PROP_VISIBLE) == -1)
    {
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);
    }
    cv::imshow(windowName, edges);
}

void sp::SubPix::drawContour(const int& contourId, const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& contoursImage, const cv::Scalar& color, bool markers)
{
    if (contourId < (int)contoursPts.size()) {

        if (markers) {
            for (int j = 0; j < contoursPts[contourId].points.size(); j += 10) {
				cv::Point2f pt(contoursPts[contourId].points[j].x, contoursPts[contourId].points[j].y);
                cv::drawMarker(contoursImage, pt, color, cv::MarkerTypes::MARKER_TILTED_CROSS, 10, 1);
            }
        }
        else {

            std::vector < std::vector< cv::Point > > contours;
            contourVec2PtsVecVecForDisplay(contoursPts, contours);

            cv::drawContours(contoursImage, contours, contourId, color, 1, 8);
        }
    }
}

void sp::SubPix::drawContours(const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& contoursImage, const std::vector<cv::Vec4i>& hierarchy)
{
    cv::RNG rng;

    std::vector < std::vector< cv::Point > > contours;
	contourVec2PtsVecVecForDisplay(contoursPts, contours);

    for (int i = 0; i < (int)contours.size(); i++) {
        cv::Scalar color(rng.uniform(1, 254), rng.uniform(1, 254), rng.uniform(1, 254));
        cv::drawContours(contoursImage, contours, i, color, 1, 8, hierarchy);
    }
}

void sp::SubPix::showContours(const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& image, cv::Mat& contours, const std::string& windowName, const std::vector<cv::Vec4i>& hierarchy, bool markers)
{
    if (!contours.empty()) {

        if (markers) {

            cv::RNG rng;

            for (int i = 0; i < (int)contoursPts.size(); ++i)
            {
                //if (i == 50)
                {
                    // set color
                    cv::Scalar color(rng.uniform(1, 254), rng.uniform(1, 254), rng.uniform(1, 254));

                    // draw contour
                    drawContour(i, contoursPts, contours, color, markers);
                }
            }
        }
        else {
            drawContours(contoursPts, contours, hierarchy);
        }

        // show contours
        if (cv::getWindowProperty(windowName, cv::WindowPropertyFlags::WND_PROP_VISIBLE) == -1) {
            cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);
        }
        cv::imshow(windowName, contours);
    }
}

void sp::SubPix::showContour(const int& contourId, const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& contours, const std::string& windowName, const cv::Scalar& color, bool marker)
{
    // draw contour
    drawContour(contourId, contoursPts, contours, color, marker);

    // show contours
    if (cv::getWindowProperty(windowName, cv::WindowPropertyFlags::WND_PROP_VISIBLE) == -1) {
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);
    }
    cv::imshow(windowName, contours);
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
    cv::destroyWindow(m_PIXEL_STATE_AFTER_CONTOURS_DETECTION_WINDOW_NAME);
    cv::destroyWindow(m_CONTOURS_WINDOW_NAME);
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

    if (cv::getWindowProperty(windowName, cv::WindowPropertyFlags::WND_PROP_VISIBLE) == -1) {
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);
    }
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

	if (cv::getWindowProperty(windowName, cv::WindowPropertyFlags::WND_PROP_VISIBLE) == -1) {
		cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);
	}
	cv::imshow(windowName, movingEdges);
	std::cout << "Displaying moving edges; press any key (in " << windowName << ") to continue.\n";

	return movingEdges;
}

cv::Mat sp::SubPix::displayImageSequenceEdgesAmbiguities(int imageWidth, int imageHeight, const std::string & windowName)
{
	std::cout << "displayImageSequenceEdgesAmbiguities..." << std::endl;

	cv::Mat ambiguityImage(imageHeight, imageWidth, CV_8UC3, cv::Scalar(0, 0, 0));
	std::map < int, int> ambiguities;

	for (std::map < cv::String, std::map< int, std::vector<sp::EdgesSubPix::Edge> > >::iterator imageListAmbiguitiesIter = m_imageListEdgesAmbiguities.begin(); imageListAmbiguitiesIter != m_imageListEdgesAmbiguities.end(); ++imageListAmbiguitiesIter)
	{
		for (std::map< int, std::vector<sp::EdgesSubPix::Edge> >::iterator iter = imageListAmbiguitiesIter->second.begin(); iter != imageListAmbiguitiesIter->second.end(); ++iter) {

			std::map < int, int>::iterator ambibuityIter = ambiguities.find(iter->first);
			if (ambibuityIter != ambiguities.end()) {
				ambibuityIter->second++;
			}
			else
			{
				ambiguities[iter->first] = 1;
			}
		}
	}

	for (std::map < int, int>::iterator ambiguitiesIter = ambiguities.begin(); ambiguitiesIter != ambiguities.end(); ++ambiguitiesIter)
	{
		int x = int(ambiguitiesIter->first / imageHeight);
		int y = int(ambiguitiesIter->first % imageHeight);

		cv::Mat_<cv::Vec3b> _frame = ambiguityImage;
		if (ambiguitiesIter->second>1)
		{
			// pixel has multi edges
			_frame(y, x)[0] = 0;
			_frame(y, x)[1] = 0;
			_frame(y, x)[2] = 255;
		}
		else {
			// pixel has only one edge
			_frame(y, x)[0] = 255;
			_frame(y, x)[1] = 255;
			_frame(y, x)[2] = 255;
		}
		ambiguityImage = _frame;
	}

	if (cv::getWindowProperty(windowName, cv::WindowPropertyFlags::WND_PROP_VISIBLE) == -1) {
		cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);
	}

	// legend
	cv::String ambiguity = "red: ambiguities on image sequence edges";
	cv::String noAmbiguity = "white: no ambiguity on image sequence edges";
	int space = 20;
	cv::putText(ambiguityImage, ambiguity, cv::Point(space, space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);
	cv::putText(ambiguityImage, noAmbiguity, cv::Point(space, 2 * space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1);

	cv::imshow(windowName, ambiguityImage);
	std::cout << "Displaying ambiguity images; press any key (in " << windowName << ") to continue.\n";

	return ambiguityImage;
}

cv::Mat sp::SubPix::displayImageSequenceContoursAmbiguities(int imageWidth, int imageHeight, const std::string & windowName)
{
	std::cout << "displayImageSequenceContoursAmbiguities..." << std::endl;

	cv::Mat ambiguityImage(imageHeight, imageWidth, CV_8UC3, cv::Scalar(0, 0, 0));
	std::map < int, int> ambiguities;

	for (std::map < cv::String, std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > > >::iterator imageListContoursAmbiguitiesIter = m_imageListContoursAmbiguities.begin(); imageListContoursAmbiguitiesIter != m_imageListContoursAmbiguities.end(); ++imageListContoursAmbiguitiesIter)
	{
		for (std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > >::iterator edgesIter = imageListContoursAmbiguitiesIter->second.begin(); edgesIter != imageListContoursAmbiguitiesIter->second.end(); ++edgesIter) {

			std::map < int, int>::iterator ambibuityIter = ambiguities.find(edgesIter->first);

			int contours = (int)edgesIter->second.size();
			if (contours > 1) {
				ambiguities[edgesIter->first] = contours;
			}
			
			for (std::map<int, std::vector<sp::EdgesSubPix::Edge> >::iterator contoursIter = edgesIter->second.begin(); contoursIter != edgesIter->second.end(); ++contoursIter) {

				std::map < int, int>::iterator ambibuityIter = ambiguities.find(edgesIter->first);
				if (ambibuityIter != ambiguities.end()) {

					std::vector<sp::EdgesSubPix::Edge> pts;
					std::copy(contoursIter->second.begin(), contoursIter->second.end(), std::back_inserter(pts));

					// remove duplicates
					auto comp = [](sp::EdgesSubPix::Edge& edge1, sp::EdgesSubPix::Edge& edge2) {  return cv::norm(edge1.point - edge2.point) < 1e-3f;  };
					auto uniquePts = std::unique(pts.begin(), pts.end(), comp);
					pts.erase(uniquePts, pts.end());		
					if (pts.size() > 1)
					{
						ambibuityIter->second++;
					}
				}
				else
				{
					ambiguities[edgesIter->first] = 1;
				}
			}
		}
	}

	for (std::map < int, int>::iterator ambiguitiesIter = ambiguities.begin(); ambiguitiesIter != ambiguities.end(); ++ambiguitiesIter)
	{
		int x = int(ambiguitiesIter->first / imageHeight);
		int y = int(ambiguitiesIter->first % imageHeight);

		cv::Mat_<cv::Vec3b> _frame = ambiguityImage;

		if (ambiguitiesIter->second>1)
		{
			// pixel has multi edges
			_frame(y, x)[0] = 0;
			_frame(y, x)[1] = 0;
			_frame(y, x)[2] = 255;
		}
		else {
			// pixel has only one edge
			_frame(y, x)[0] = 255;
			_frame(y, x)[1] = 255;
			_frame(y, x)[2] = 255;
		}
		ambiguityImage = _frame;
	}

	if (cv::getWindowProperty(windowName, cv::WindowPropertyFlags::WND_PROP_VISIBLE) == -1) {
		cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);
	}

	// legend
	cv::String ambiguity = "red: ambiguities on image sequence contours edges";
	cv::String noAmbiguity = "white: no ambiguity on image sequence contours edges";
	int space = 20;
	cv::putText(ambiguityImage, ambiguity, cv::Point(space, space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);
	cv::putText(ambiguityImage, noAmbiguity, cv::Point(space, 2 * space), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1);

	cv::imshow(windowName, ambiguityImage);
	std::cout << "Displaying ambiguity images; press any key (in " << windowName << ") to continue.\n";

	return ambiguityImage;
}
