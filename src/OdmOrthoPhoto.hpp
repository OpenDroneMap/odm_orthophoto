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

// GDAL
#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()

#include <Eigen/Dense>

// Logger
#include "Logger.hpp"

#include <filesystem>
namespace fs = std::filesystem;

struct Bounds{
    float xMin;
    float xMax;
    float yMin;
    float yMax;

    Bounds() : xMin(0), xMax(0), yMin(0), yMax(0) {}
    Bounds(float xMin, float xMax, float yMin, float yMax) :
        xMin(xMin), xMax(xMax), yMin(yMin), yMax(yMax){}
    Bounds(const Bounds &b) {
        xMin = b.xMin;
        xMax = b.xMax;
        yMin = b.yMin;
        yMax = b.yMax;
    }
};

typedef Eigen::Vector3f PointXYZ;
typedef Eigen::Vector2f Tex2D;

struct Face{
    PointXYZ v1;
    PointXYZ v2;
    PointXYZ v3;
    Tex2D t1;
    Tex2D t2;
    Tex2D t3;
};

struct TextureMesh{
    std::vector<PointXYZ> vertices;
    std::vector<Tex2D> uvs;
    std::unordered_map<std::string, cv::Mat> materials;
    std::unordered_map<std::string, std::vector<Face> > faces;

};

/*!
 * \brief   The OdmOrthoPhoto class is used to create an orthographic photo over a given area.
 *          The class reads an oriented textured mesh from an OBJ-file.
 *          The class uses file read from pcl.
 *          The class uses image read and write from opencv.
 */
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
    Eigen::Transform<float, 3, Eigen::Affine> getROITransform(float xMin, float yMin) const;

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
      *
      * \param texture The texture of the polygon.
      * \param polygon The polygon as athree indices relative meshCloud.
      * \param meshCloud Contains all vertices.
      * \param uvs Contains the texture coordinates for the active material.
      * \param faceIndex The index of the face.
      */
    //template <typename T>
    //void drawTexturedTriangle(const cv::Mat &texture, const pcl::Vertices &polygon, const pcl::PointCloud<pcl::PointXYZ>::Ptr &meshCloud, const std::vector<Eigen::Vector2f> &uvs, size_t faceIndex);

    /*!
      * \brief Sets the color of a pixel in the photo.
      *
      * \param row The row index of the pixel.
      * \param col The column index of the pixel.
      * \param s The u texture-coordinate, multiplied with the number of columns in the texture.
      * \param t The v texture-coordinate, multiplied with the number of rows in the texture.
      * \param texture The texture from which to get the color.
      **/
    //template <typename T>
    //void renderPixel(int row, int col, float u, float v, const cv::Mat &texture);

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
    //void getBarycentricCoordinates(PointXYZ v1, PointXYZ v2, PointXYZ v3, float x, float y, float &l1, float &l2, float &l3) const;

    //bool isSliverPolygon(PointXYZ v1, PointXYZ v2, PointXYZ v3) const;
    void loadObjFile(std::string inputFile, TextureMesh &mesh);

    Logger          log_;               /**< Logging object. */

    std::vector<std::string> inputFiles;
    std::string     outputFile_;        /**< Path to the destination file. */
    std::string     outputCornerFile_;  /**< Path to the output corner file. */
    std::string     logFile_;           /**< Path to the log file. */
    std::string     bandsOrder;

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
