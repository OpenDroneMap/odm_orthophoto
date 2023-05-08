#pragma once

// C++
#include <limits.h>
#include <istream>
#include <ostream>
#include <vector>
#include <unordered_map>

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/photo.hpp>

// GDAL
#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()

#include <Eigen/Dense>

// Logger
#include "Logger.hpp"

#include <filesystem>
namespace fs = std::filesystem;

struct Bounds{
    double xMin;
    double xMax;
    double yMin;
    double yMax;

    Bounds() : xMin(0), xMax(0), yMin(0), yMax(0) {}
    Bounds(double xMin, double xMax, double yMin, double yMax) :
        xMin(xMin), xMax(xMax), yMin(yMin), yMax(yMax){}
    Bounds(const Bounds &b) {
        xMin = b.xMin;
        xMax = b.xMax;
        yMin = b.yMin;
        yMax = b.yMax;
    }
    constexpr Bounds& operator=(const Bounds& other){
        xMin = other.xMin;
        xMax = other.xMax;
        yMin = other.yMin;
        yMax = other.yMax;
        return *this;
    }
};

typedef Eigen::Vector3d PointXYZ;
typedef Eigen::Vector2f Tex2D;

struct Face{
    size_t v1i;
    size_t v2i;
    size_t v3i;
    size_t t1i;
    size_t t2i;
    size_t t3i;
};

struct TextureMesh{
    std::vector<PointXYZ> vertices;
    std::vector<Tex2D> uvs;
    std::unordered_map<std::string, cv::Mat> materials;
    std::unordered_map<std::string, std::vector<Face> > faces;
    std::vector<std::string> materials_idx;
};

class OdmOrthoPhoto
{
public:
    OdmOrthoPhoto();
    ~OdmOrthoPhoto();

    /*!
     * \brief   run     Runs the ortho photo functionality using the provided input arguments.
     *                  For a list of accepted arguments, pleas see the main page documentation or
     *                  call the program with parameter "-help".
     * \param   argc    Application argument count.
     * \param   argv    Argument values.
     * \return  0       if successful.
     */
    int run(int argc, char* argv[]);

private:
    int width, height;
    Bounds bounds;

    void parseArguments(int argc, char* argv[]);
    void printHelp();

    void createOrthoPhoto();

    /*!
      * \brief Compute the boundary points so that the entire model fits inside the photo.
      *
      * \param mesh The model which decides the boundary.
      */
    Bounds computeBoundsForModel(const TextureMesh &mesh);
    
    /*!
      * \brief Creates a transformation which aligns the area for the orthophoto.
      */
    Eigen::Transform<double, 3, Eigen::Affine> getROITransform(double xMin, double yMin) const;

    template <typename T>
    void initBands(int count);

    template <typename T>
    void initAlphaBand();

    template <typename T>
    void finalizeAlphaBand();

    void saveTIFF(const std::string &filename, GDALDataType dataType);
    
    /*!
      * \brief Renders a triangle into the ortho photo.
      *
      *        Pixel center defined as middle of pixel for triangle rasterisation, and in lower left corner for texture look-up.
      */
    template <typename T>
    void drawTexturedTriangle(const cv::Mat &texture, const TextureMesh &mesh, Face &face);

    /*!
      * \brief Sets the color of a pixel in the photo.
      *
      * \param row The row index of the pixel.
      * \param col The column index of the pixel.
      * \param s The u texture-coordinate, multiplied with the number of columns in the texture.
      * \param t The v texture-coordinate, multiplied with the number of rows in the texture.
      * \param texture The texture from which to get the color.
      **/
    template <typename T>
    void renderPixel(int row, int col, double u, double v, const cv::Mat &texture);

    /*!
      * \brief Calculates the barycentric coordinates of a point in a triangle.
      *
      * \param v1 The first triangle vertex.
      * \param v2 The second triangle vertex.
      * \param v3 The third triangle vertex.
      * \param x The x coordinate of the point.
      * \param y The y coordinate of the point.
      * \param l1 The first vertex weight.
      * \param l2 The second vertex weight.
      * \param l3 The third vertex weight.
      */
    void getBarycentricCoordinates(const PointXYZ &v1, const PointXYZ &v2, const PointXYZ &v3, double x, double y, double &l1, double &l2, double &l3) const;

    bool isSliverPolygon(const PointXYZ &v1, const PointXYZ &v2, const PointXYZ &v3) const;
    void loadObjFile(std::string inputFile, TextureMesh &mesh);

    template <typename T>
    void inpaint(float threshold, int CV_TYPE);
    Logger          log_;               /**< Logging object. */

    std::vector<std::string> inputFiles;
    std::string     outputFile_;        /**< Path to the destination file. */
    std::string     outputCornerFile_;  /**< Path to the output corner file. */
    std::string     logFile_;           /**< Path to the log file. */
    std::string     bandsOrder;
    float inpaintThreshold;
    int outputDepthIdx;

    std::vector<std::pair<std::string, std::string> > coOptions;
    std::vector<std::pair<std::string, std::string> > gdalConfigs;
    std::string aSrs;
    double utm_east_offset;
    double utm_north_offset;

    float           resolution_;        /**< The number of pixels per meter in the ortho photo. */

    std::vector<void *>    bands;
    std::vector<GDALColorInterp> colorInterps;
    std::vector<std::string> bandDescriptions;
    void *alphaBand; // Keep alpha band separate
    int currentBandIndex;

    cv::Mat         depth_;             /**< The depth of the ortho photo as an OpenCV matrix, CV_32F. */

};

class OdmOrthoPhotoException : public std::exception
{

public:
    OdmOrthoPhotoException() : message("Error in OdmOrthoPhoto") {}
    OdmOrthoPhotoException(std::string msgInit) : message("Error in OdmOrthoPhoto:\n" + msgInit) {}
    ~OdmOrthoPhotoException() throw() {}
    virtual const char* what() const throw() {return message.c_str(); }

private:
    std::string message;
};

// Utils

static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

static inline std::vector<std::string> split(const std::string &s, const std::string &delimiter){
    size_t posStart = 0, posEnd, delimLen = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((posEnd = s.find(delimiter, posStart)) != std::string::npos) {
        token = s.substr(posStart, posEnd - posStart);
        posStart = posEnd + delimLen;
        res.push_back(token);
    }

    res.push_back(s.substr(posStart));
    return res;
}
