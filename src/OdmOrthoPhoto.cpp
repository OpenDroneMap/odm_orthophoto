#include <math.h>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <string>

#include "OdmOrthoPhoto.hpp"

OdmOrthoPhoto::OdmOrthoPhoto()
    :log_(false){
    outputFile_ = "ortho.tif";
    logFile_    = "log.txt";
    outputCornerFile_ = "";
    bandsOrder = "red,green,blue";

    resolution_ = 0.0f;

    alphaBand = nullptr;
    currentBandIndex = 0;
}

OdmOrthoPhoto::~OdmOrthoPhoto()
{
}

int OdmOrthoPhoto::run(int argc, char *argv[])
{
    try
    {
        parseArguments(argc, argv);
        createOrthoPhoto();
    }
    catch (const OdmOrthoPhotoException& e)
    {
        log_.setIsPrintingInCout(true);
        log_ << e.what() << "\n";
        log_.print(logFile_);
        return EXIT_FAILURE;
    }
    catch (const std::exception& e)
    {
        log_.setIsPrintingInCout(true);
        log_ << "Error in OdmOrthoPhoto:\n";
        log_ << e.what() << "\n";
        log_.print(logFile_);
        return EXIT_FAILURE;
    }
    catch (...)
    {
        log_.setIsPrintingInCout(true);
        log_ << "Unknown error, terminating:\n";
        log_.print(logFile_);
        return EXIT_FAILURE;
    }
    
    log_.print(logFile_);
    
    return EXIT_SUCCESS;
}

void OdmOrthoPhoto::parseArguments(int argc, char *argv[])
{
    logFile_ = std::string(argv[0]) + "_log.txt";
    log_ << logFile_ << "\n\n";
    
    // If no arguments were passed, print help.
    if (argc == 1)
    {
        printHelp();
    }
    
    log_ << "Arguments given\n";
    for(int argIndex = 1; argIndex < argc; ++argIndex)
    {
        log_ << argv[argIndex] << '\n';
    }
    
    log_ << '\n';
    for(int argIndex = 1; argIndex < argc; ++argIndex)
    {
        // The argument to be parsed.
        std::string argument = std::string(argv[argIndex]);
        
        if(argument == "-help")
        {
            printHelp();
        }
        else if(argument == "-resolution")
        {
            ++argIndex;
            if (argIndex >= argc)
            {
                throw OdmOrthoPhotoException("Argument '" + argument + "' expects 1 more input following it, but no more inputs were provided.");
            }
            std::stringstream ss(argv[argIndex]);
            ss >> resolution_;
            log_ << "Resolution count was set to: " << resolution_ << "pixels/meter\n";
        }
        else if(argument == "-verbose")
        {
            log_.setIsPrintingInCout(true);
        }
        else if (argument == "-logFile")
        {
            ++argIndex;
            if (argIndex >= argc)
            {
                throw OdmOrthoPhotoException("Missing argument for '" + argument + "'.");
            }
            logFile_ = std::string(argv[argIndex]);
            std::ofstream testFile(logFile_.c_str());
            if (!testFile.is_open())
            {
                throw OdmOrthoPhotoException("Argument '" + argument + "' has a bad value.");
            }
            log_ << "Log file path was set to: " << logFile_ << "\n";
        }
        else if(argument == "-inputFiles")
        {
            argIndex++;
            if (argIndex >= argc)
            {
                throw OdmOrthoPhotoException("Argument '" + argument + "' expects 1 more input following it, but no more inputs were provided.");
            }
            std::string inputFilesArg = std::string(argv[argIndex]);
            std::stringstream ss(inputFilesArg);
            std::string item;
            while(std::getline(ss, item, ',')){
                inputFiles.push_back(item);
            }
        }
        else if(argument == "-bands")
        {
            argIndex++;
            if (argIndex >= argc)
            {
                throw OdmOrthoPhotoException("Argument '" + argument + "' expects 1 more input following it, but no more inputs were provided.");
            }
            bandsOrder = std::string(argv[argIndex]);
        }
        else if(argument == "-outputFile")
        {
            argIndex++;
            if (argIndex >= argc)
            {
                throw OdmOrthoPhotoException("Argument '" + argument + "' expects 1 more input following it, but no more inputs were provided.");
            }
            outputFile_ = std::string(argv[argIndex]);
            log_ << "Writing output to: " << outputFile_ << "\n";
        }
        else if(argument == "-outputCornerFile")
        {
            argIndex++;
            if (argIndex >= argc)
            {
                throw OdmOrthoPhotoException("Argument '" + argument + "' expects 1 more input following it, but no more inputs were provided.");
            }
            outputCornerFile_ = std::string(argv[argIndex]);
        }
        else
        {
            printHelp();
            throw OdmOrthoPhotoException("Unrecognised argument '" + argument + "'");
        }
    }
    log_ << "\n";

    std::stringstream ss(bandsOrder);
    std::string item;
    while(std::getline(ss, item, ',')){
        std::string itemL = item;
        // To lower case
        std::transform(itemL.begin(), itemL.end(), itemL.begin(), [](unsigned char c){ return std::tolower(c); });

        if (itemL == "red" || itemL == "r"){
            colorInterps.push_back(GCI_RedBand);
        }else if (itemL == "green" || itemL == "g"){
            colorInterps.push_back(GCI_GreenBand);
        }else if (itemL == "blue" || itemL == "b"){
            colorInterps.push_back(GCI_BlueBand);
        }else{
            colorInterps.push_back(GCI_GrayIndex);
        }
        bandDescriptions.push_back(item);
    }
}

void OdmOrthoPhoto::printHelp()
{
    log_.setIsPrintingInCout(true);

    log_ << "odm_orthophoto\n\n";

    log_ << "Purpose\n";
    log_ << "Create an orthograpical photo from an oriented textured mesh.\n\n";

    log_ << "Usage:\n";
    log_ << "The program requires a path to an input OBJ mesh file and a resolution, as pixels/m. All other input parameters are optional.\n\n";

    log_ << "The following flags are available\n";
    log_ << "Call the program with flag \"-help\", or without parameters to print this message, or check any generated log file.\n";
    log_ << "Call the program with flag \"-verbose\", to print log messages in the standard output stream as well as in the log file.\n\n";

    log_ << "Parameters are specified as: \"-<argument name> <argument>\", (without <>), and the following parameters are configureable:\n";
    log_ << "\"-inputFiles <path>[,<path2>,<path3>,...]\" (mandatory)\n";
    log_ << "\"Input obj files that must contain a textured mesh.\n\n";

    log_ << "\"-outputFile <path>\" (optional, default: ortho.jpg)\n";
    log_ << "\"Target file in which the orthophoto is saved.\n\n";

    log_ << "\"-outputCornerFile <path>\" (optional)\n";
    log_ << "\"Target text file for boundary corner points, written as \"xmin ymin xmax ymax\".\n\n";

    log_ << "\"-resolution <pixels/m>\" (mandatory)\n";
    log_ << "\"The number of pixels used per meter.\n\n";

    log_ << "\"-bands red,green,blue,[...]\" (optional)\n";
    log_ << "\"Naming of bands to assign color interpolation values when creating output TIFF.\n\n";

    log_.setIsPrintingInCout(false);
}

void OdmOrthoPhoto::saveTIFF(const std::string &filename, GDALDataType dataType){
    GDALAllRegister();
    GDALDriverH hDriver = GDALGetDriverByName( "GTiff" );
    if (!hDriver){
        std::cerr << "Cannot initialize GeoTIFF driver. Check your GDAL installation." << std::endl;
        exit(1);
    }
    char **papszOptions = NULL;
    GDALDatasetH hDstDS = GDALCreate( hDriver, filename.c_str(), width, height,
                                      static_cast<int>(bands.size() + 1), dataType, papszOptions );
    GDALRasterBandH hBand;

    // Bands
    size_t i = 0;
    for (; i < bands.size(); i++){
        hBand = GDALGetRasterBand( hDstDS, static_cast<int>(i) + 1 );

        GDALColorInterp interp = GCI_GrayIndex;
        if (i < colorInterps.size()){
            interp = colorInterps[i];
        }
        GDALSetRasterColorInterpretation(hBand, interp );

        if (i < bandDescriptions.size()){
            GDALSetDescription(hBand, bandDescriptions[i].c_str());
        }

        if (GDALRasterIO( hBand, GF_Write, 0, 0, width, height,
                    bands[i], width, height, dataType, 0, 0 ) != CE_None){
            std::cerr << "Cannot write TIFF to " << filename << std::endl;
            exit(1);
        }
    }

    // Alpha
    if (dataType == GDT_Float32){
        finalizeAlphaBand<float>();
    }else if (dataType == GDT_UInt16){
        finalizeAlphaBand<uint16_t>();
    }else if (dataType == GDT_Byte){
        finalizeAlphaBand<uint8_t>();
    }else{
        throw OdmOrthoPhotoException("Invalid data type");
    }

    // Alpha
    hBand = GDALGetRasterBand( hDstDS, static_cast<int>(i) + 1 );

    // Set alpha band
    GDALSetRasterColorInterpretation(hBand, GCI_AlphaBand );

    if (GDALRasterIO( hBand, GF_Write, 0, 0, width, height,
                alphaBand, width, height, dataType, 0, 0 ) != CE_None){
        std::cerr << "Cannot write TIFF (alpha) to " << filename << std::endl;
        exit(1);
    }

    GDALClose( hDstDS );
}

template <typename T>
inline T maxRange(){
    return static_cast<T>(pow(2, sizeof(T) * 8) - 1);
}

template <typename T>
void OdmOrthoPhoto::initBands(int count){
    size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);

    // Channels
    T initValue = maxRange<T>();

    for (int i = 0; i < count; i++){
        T *arr = new T[pixelCount];
        for (size_t j = 0; j < pixelCount; j++){
            arr[j] = initValue;
        }
        bands.push_back(static_cast<void *>(arr));
    }
}

template <typename T>
void OdmOrthoPhoto::initAlphaBand(){
     size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
     // Alpha
     if (alphaBand == nullptr){
         T *arr = new T[pixelCount];
         for (size_t j = 0; j < pixelCount; j++){
             arr[j] = 0.0;
         }
         alphaBand = static_cast<void *>(arr);
     }
}

template <typename T>
void OdmOrthoPhoto::finalizeAlphaBand(){
     // Adjust alpha band values, only pixels that have
     // values on all bands should be visible

     size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
     int channels = bands.size();

     T *arr = reinterpret_cast<T *>(alphaBand);
     for (size_t j = 0; j < pixelCount; j++){
         arr[j] = arr[j] >= channels ? 255.0 : 0.0;
     }
}


void OdmOrthoPhoto::createOrthoPhoto()
{
    if(inputFiles.size() == 0)
    {
        throw OdmOrthoPhotoException("Failed to create ortho photo, no texture meshes given.");
    }

    int textureDepth = -1;
    bool primary = true;
    Bounds bounds;

    for (auto &inputFile : inputFiles){
        log_ << "Reading mesh file... " << inputFile << "\n";

        TextureMesh mesh;
        loadObjFile(inputFile, mesh);


    }} // TODO REMOVE

    /*
        log_ << "Mesh file read.\n\n";

        Bounds b = computeBoundsForModel(mesh);

        log_ << "Model bounds x : " << b.xMin << " -> " << b.xMax << '\n';
        log_ << "Model bounds y : " << b.yMin << " -> " << b.yMax << '\n';

        if (primary){
            bounds = b;
        }else{
            // Quick check
            if (b.xMin != bounds.xMin ||
                    b.xMax != bounds.xMax ||
                    b.yMin != bounds.yMin ||
                    b.yMax != bounds.yMax){
                throw OdmOrthoPhotoException("Bounds between models must all match, but they don't.");
            }
        }

        // The size of the area.
        float xDiff = bounds.xMax - bounds.xMin;
        float yDiff = bounds.yMax - bounds.yMin;
        log_ << "Model area : " << xDiff*yDiff << "m2\n";

        // The resolution necessary to fit the area with the given resolution.
        height = static_cast<int>(std::ceil(resolution_*yDiff));
        width = static_cast<int>(std::ceil(resolution_*xDiff));

        depth_ = cv::Mat::zeros(height, width, CV_32F) - std::numeric_limits<float>::infinity();
        log_ << "Model resolution, width x height : " << width << "x" << height << '\n';

        // Check size of photo.
        if(0 >= height*width)
        {
            if(0 >= height)
            {
                log_ << "Warning: ortho photo has zero area, height = " << height << ". Forcing height = 1.\n";
                height = 1;
            }
            if(0 >= width)
            {
                log_ << "Warning: ortho photo has zero area, width = " << width << ". Forcing width = 1.\n";
                width = 1;
            }
            log_ << "New ortho photo resolution, width x height : " << width << "x" << height << '\n';
        }

        // Contains the vertices of the mesh.
        pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromPCLPointCloud2 (mesh.cloud, *meshCloud);

        // Creates a transformation which aligns the area for the ortho photo.
        Eigen::Transform<float, 3, Eigen::Affine> transform = getROITransform(bounds.xMin, -bounds.yMax);
        log_ << "Translating and scaling mesh...\n";

        // Move the mesh into position.
        pcl::transformPointCloud(*meshCloud, *meshCloud, transform);
        log_ << ".. mesh translated and scaled.\n\n";

        // Flatten texture coordinates.
        std::vector<Eigen::Vector2f> uvs;
        uvs.reserve(mesh.tex_coordinates.size());
        for(size_t t = 0; t < mesh.tex_coordinates.size(); ++t)
        {
            uvs.insert(uvs.end(), mesh.tex_coordinates[t].begin(), mesh.tex_coordinates[t].end());
        }

        // The current material texture
        cv::Mat texture;

        // Used to keep track of the global face index.
        size_t faceOff = 0;

        log_ << "Rendering the ortho photo...\n";

        // Iterate over each part of the mesh (one per material).
        for(size_t t = 0; t < mesh.tex_materials.size(); ++t)
        {
            // The material of the current submesh.
            pcl::TexMaterial material = mesh.tex_materials[t];
            texture = cv::imread(material.tex_file, cv::IMREAD_ANYDEPTH | cv::IMREAD_UNCHANGED);

            // BGR to RGB when necessary
            if (texture.channels() == 3){
                cv::cvtColor(texture, texture, cv::COLOR_BGR2RGB);
            }

            // The first material determines the bit depth
            // Init ortho photo
            if (t == 0){
                if (primary) textureDepth = texture.depth();
                else if (textureDepth != texture.depth()){
                    // Try to convert
                    if (textureDepth == CV_8U){
                        if (texture.depth() == CV_16U){
                            // 16U to 8U
                            texture.convertTo(texture, CV_8U, 255.0f / 65535.0f);
                        }else{
                            throw OdmOrthoPhotoException("Unknown conversion from CV_8U");
                        }
                    }else if (textureDepth == CV_16U){
                        if (texture.depth() == CV_8U){
                            // 8U to 16U
                            texture.convertTo(texture, CV_16U, 65535.0f / 255.0f);
                        }else{
                            throw OdmOrthoPhotoException("Unknown conversion from CV_16U");
                        }
                    }else{
                         throw OdmOrthoPhotoException("Texture depth is not the same for all models and could not be converted");
                    }
                }

                log_ << "Texture channels: " << texture.channels() << "\n";

                try{
                    if (textureDepth == CV_8U){
                        log_ << "Texture depth: 8bit\n";
                        initBands<uint8_t>(texture.channels());
                        if (primary) initAlphaBand<uint8_t>();
                    }else if (textureDepth == CV_16U){
                        log_ << "Texture depth: 16bit\n";
                        initBands<uint16_t>(texture.channels());
                        if (primary) initAlphaBand<uint16_t>();
                    }else if (textureDepth == CV_32F){
                        log_ << "Texture depth: 32bit (float)\n";
                        initBands<float>(texture.channels());
                        if (primary) initAlphaBand<float>();
                    }else{
                        std::cerr << "Unsupported bit depth value: " << textureDepth;
                        exit(1);
                    }
                }catch(const std::bad_alloc &){
                    std::cerr << "Couldn't allocate enough memory to render the orthophoto (" << width << "x" << height << " cells = " << ((long long)width * (long long)height * 4) << " bytes). Try to increase the --orthophoto-resolution parameter to a larger integer or add more RAM.\n";
                    exit(1);
                }
            }

            // Check for missing files.
            if(texture.empty())
            {
                log_ << "Material texture could not be read:\n";
                log_ << material.tex_file << '\n';
                log_ << "Could not be read as image, does the file exist?\n";
                continue; // Skip to next material.
            }

            // The faces of the current submesh.
            std::vector<pcl::Vertices> faces = mesh.tex_polygons[t];

            // Iterate over each face...
            for(size_t faceIndex = 0; faceIndex < faces.size(); ++faceIndex)
            {
                // The current polygon.
                pcl::Vertices polygon = faces[faceIndex];

                // ... and draw it into the ortho photo.
                if (textureDepth == CV_8U){
                    drawTexturedTriangle<uint8_t>(texture, polygon, meshCloud, uvs, faceIndex+faceOff);
                }else if (textureDepth == CV_16U){
                    drawTexturedTriangle<uint16_t>(texture, polygon, meshCloud, uvs, faceIndex+faceOff);
                }else if (textureDepth == CV_32F){
                    drawTexturedTriangle<float>(texture, polygon, meshCloud, uvs, faceIndex+faceOff);
                }
            }
            faceOff += faces.size();
            log_ << "Material " << t << " rendered.\n";
        }
        log_ << "... model rendered\n";

        currentBandIndex += texture.channels();
        primary = false;
    }

    log_ << '\n';
    log_ << "Writing ortho photo to " << outputFile_ << "\n";

    if (textureDepth == CV_8U){
        saveTIFF(outputFile_, GDT_Byte);
    }else if (textureDepth == CV_16U){
        saveTIFF(outputFile_, GDT_UInt16);
    }else if (textureDepth == CV_32F){
        saveTIFF(outputFile_, GDT_Float32);
    }else{
        std::cerr << "Unsupported bit depth value: " << textureDepth;
        exit(1);
    }

    if (!outputCornerFile_.empty())
    {
        log_ << "Writing corner coordinates to " << outputCornerFile_ << "\n";
        std::ofstream cornerStream(outputCornerFile_.c_str());
        if (!cornerStream.is_open())
        {
            throw OdmOrthoPhotoException("Failed opening output corner file " + outputCornerFile_ + ".");
        }
        cornerStream.setf(std::ios::scientific, std::ios::floatfield);
        cornerStream.precision(17);
        cornerStream << bounds.xMin << " " << bounds.yMin << " " << bounds.xMax << " " << bounds.yMax;
        cornerStream.close();
    }

    log_ << "Orthophoto generation done.\n";
}

Bounds OdmOrthoPhoto::computeBoundsForModel(const pcl::TextureMesh &mesh)
{
    log_ << "Set boundary to contain entire model.\n";

    // The boundary of the model.
    Bounds r;

    r.xMin = std::numeric_limits<float>::infinity();
    r.xMax = -std::numeric_limits<float>::infinity();
    r.yMin = std::numeric_limits<float>::infinity();
    r.yMax = -std::numeric_limits<float>::infinity();

    // Contains the vertices of the mesh.
    pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2 (mesh.cloud, *meshCloud);

    for (size_t i = 0; i < meshCloud->points.size(); i++){
        pcl::PointXYZ v = meshCloud->points[i];
        r.xMin = std::min(r.xMin, v.x);
        r.xMax = std::max(r.xMax, v.x);
        r.yMin = std::min(r.yMin, v.y);
        r.yMax = std::max(r.yMax, v.y);
    }

    log_ << "Boundary points:\n";
    log_ << "Point 1: " << r.xMin << " " << r.yMin << "\n";
    log_ << "Point 2: " << r.xMin << " " << r.yMax << "\n";
    log_ << "Point 3: " << r.xMax << " " << r.yMax << "\n";
    log_ << "Point 4: " << r.xMax << " " << r.yMin << "\n";
    log_ << "\n";

    return r;
}

Eigen::Transform<float, 3, Eigen::Affine> OdmOrthoPhoto::getROITransform(float xMin, float yMin) const
{
    // The transform used to move the chosen area into the ortho photo.
    Eigen::Transform<float, 3, Eigen::Affine> transform;

    transform(0, 0) = resolution_;     // x Scaling.
    transform(1, 0) = 0.0f;
    transform(2, 0) = 0.0f;
    transform(3, 0) = 0.0f;

    transform(0, 1) = 0.0f;
    transform(1, 1) = -resolution_;     // y Scaling, mirrored for easier rendering.
    transform(2, 1) = 0.0f;
    transform(3, 1) = 0.0f;

    transform(0, 2) = 0.0f;
    transform(1, 2) = 0.0f;
    transform(2, 2) = 1.0f;
    transform(3, 2) = 0.0f;

    transform(0, 3) = -xMin * resolution_;    // x Translation
    transform(1, 3) = -yMin * resolution_;    // y Translation
    transform(2, 3) = 0.0f;
    transform(3, 3) = 1.0f;

    return transform;
}

template <typename T>
void OdmOrthoPhoto::drawTexturedTriangle(const cv::Mat &texture, const pcl::Vertices &polygon, const pcl::PointCloud<pcl::PointXYZ>::Ptr &meshCloud, const std::vector<Eigen::Vector2f> &uvs, size_t faceIndex)
{
    // The index to the vertices of the polygon.
    size_t v1i = polygon.vertices[0];
    size_t v2i = polygon.vertices[1];
    size_t v3i = polygon.vertices[2];

    // The polygon's points.
    pcl::PointXYZ v1 = meshCloud->points[v1i];
    pcl::PointXYZ v2 = meshCloud->points[v2i];
    pcl::PointXYZ v3 = meshCloud->points[v3i];

    if(isSliverPolygon(v1, v2, v3))
    {
        log_ << "Warning: Sliver polygon found at face index " << faceIndex << '\n';
        return;
    }

    // The face data. Position v*{x,y,z}. Texture coordinate v*{u,v}. * is the vertex number in the polygon.
    float v1x, v1y, v1z, v1u, v1v;
    float v2x, v2y, v2z, v2u, v2v;
    float v3x, v3y, v3z, v3u, v3v;

    // Barycentric coordinates of the currently rendered point.
    float l1, l2, l3;

    // The size of the photo, as float.
    float fRows, fCols;
    fRows = static_cast<float>(texture.rows);
    fCols = static_cast<float>(texture.cols);

    // Get vertex position.
    v1x = v1.x; v1y = v1.y; v1z = v1.z;
    v2x = v2.x; v2y = v2.y; v2z = v2.z;
    v3x = v3.x; v3y = v3.y; v3z = v3.z;

    // Get texture coordinates. 
    v1u = uvs[3*faceIndex][0]; v1v = uvs[3*faceIndex][1];
    v2u = uvs[3*faceIndex+1][0]; v2v = uvs[3*faceIndex+1][1];
    v3u = uvs[3*faceIndex+2][0]; v3v = uvs[3*faceIndex+2][1];

    // Check bounding box overlap.
    int xMin = static_cast<int>(std::min(std::min(v1x, v2x), v3x));
    if(xMin > width)
    {
        return; // Completely outside to the right.
    }
    int xMax = static_cast<int>(std::max(std::max(v1x, v2x), v3x));
    if(xMax < 0)
    {
        return; // Completely outside to the left.
    }
    int yMin = static_cast<int>(std::min(std::min(v1y, v2y), v3y));
    if(yMin > height)
    {
        return; // Completely outside to the top.
    }
    int yMax = static_cast<int>(std::max(std::max(v1y, v2y), v3y));
    if(yMax < 0)
    {
        return; // Completely outside to the bottom.
    }

    // Top point row and column positions
    float topR, topC;
    // Middle point row and column positions
    float midR, midC;
    // Bottom point row and column positions
    float botR, botC;

    // Find top, middle and bottom points.
    if(v1y < v2y)
    {
        if(v1y < v3y)
        {
            if(v2y < v3y)
            {
                // 1 -> 2 -> 3
                topR = v1y; topC = v1x;
                midR = v2y; midC = v2x;
                botR = v3y; botC = v3x;
            }
            else
            {
                // 1 -> 3 -> 2
                topR = v1y; topC = v1x;
                midR = v3y; midC = v3x;
                botR = v2y; botC = v2x;
            }
        }
        else
        {
            // 3 -> 1 -> 2
            topR = v3y; topC = v3x;
            midR = v1y; midC = v1x;
            botR = v2y; botC = v2x;
        }        
    }
    else // v2y <= v1y
    {
        if(v2y < v3y)
        {
            if(v1y < v3y)
            {
                // 2 -> 1 -> 3
                topR = v2y; topC = v2x;
                midR = v1y; midC = v1x;
                botR = v3y; botC = v3x;
            }
            else
            {
                // 2 -> 3 -> 1
                topR = v2y; topC = v2x;
                midR = v3y; midC = v3x;
                botR = v1y; botC = v1x;
            }
        }
        else
        {
            // 3 -> 2 -> 1
            topR = v3y; topC = v3x;
            midR = v2y; midC = v2x;
            botR = v1y; botC = v1x;
        }
    }

    // General appreviations:
    // ---------------------
    // tm : Top(to)Middle.
    // mb : Middle(to)Bottom.
    // tb : Top(to)Bottom.
    // c  : column.
    // r  : row.
    // dr : DeltaRow, step value per row.

    // The step along column for every step along r. Top to middle.
    float ctmdr;
    // The step along column for every step along r. Top to bottom.
    float ctbdr;
    // The step along column for every step along r. Middle to bottom.
    float cmbdr;

    ctbdr = (botC-topC)/(botR-topR);

    // The current column position, from top to middle.
    float ctm = topC;
    // The current column position, from top to bottom.
    float ctb = topC;

    // Check for vertical line between middle and top.
    if(FLT_EPSILON < midR-topR)
    {
        ctmdr = (midC-topC)/(midR-topR);

        // The first pixel row for the bottom part of the triangle.
        int rqStart = std::max(static_cast<int>(std::floor(topR+0.5f)), 0);
        // The last pixel row for the top part of the triangle.
        int rqEnd = std::min(static_cast<int>(std::floor(midR+0.5f)), height);

        // Traverse along row from top to middle.
        for(int rq = rqStart; rq < rqEnd; ++rq)
        {
            // Set the current column positions.
            ctm = topC + ctmdr*(static_cast<float>(rq)+0.5f-topR);
            ctb = topC + ctbdr*(static_cast<float>(rq)+0.5f-topR);

            // The first pixel column for the current row.
            int cqStart = std::max(static_cast<int>(std::floor(0.5f+std::min(ctm, ctb))), 0);
            // The last pixel column for the current row.
            int cqEnd = std::min(static_cast<int>(std::floor(0.5f+std::max(ctm, ctb))), width);

            for(int cq = cqStart; cq < cqEnd; ++cq)
            {
                // Get barycentric coordinates for the current point.
                getBarycentricCoordinates(v1, v2, v3, static_cast<float>(cq)+0.5f, static_cast<float>(rq)+0.5f, l1, l2, l3);

                if(0.f > l1 || 0.f > l2 || 0.f > l3)
                {
                    //continue;
                }
                
                // The z value for the point.
                float z = v1z*l1+v2z*l2+v3z*l3;

                // Check depth
                float depthValue = depth_.at<float>(rq, cq);
                if(z < depthValue)
                {
                    // Current is behind another, don't draw.
                    continue;
                }

                // The uv values of the point.
                float u, v;
                u = v1u*l1+v2u*l2+v3u*l3;
                v = v1v*l1+v2v*l2+v3v*l3;
                
                renderPixel<T>(rq, cq, u*fCols, (1.0f-v)*fRows, texture);
                
                // Update depth buffer.
                depth_.at<float>(rq, cq) = z;
            }
        }
    }

    if(FLT_EPSILON < botR-midR)
    {
        cmbdr = (botC-midC)/(botR-midR);

        // The current column position, from middle to bottom.
        float cmb = midC;

        // The first pixel row for the bottom part of the triangle.
        int rqStart = std::max(static_cast<int>(std::floor(midR+0.5f)), 0);
        // The last pixel row for the bottom part of the triangle.
        int rqEnd = std::min(static_cast<int>(std::floor(botR+0.5f)), height);

        // Traverse along row from middle to bottom.
        for(int rq = rqStart; rq < rqEnd; ++rq)
        {
            // Set the current column positions.
            ctb = topC + ctbdr*(static_cast<float>(rq)+0.5f-topR);
            cmb = midC + cmbdr*(static_cast<float>(rq)+0.5f-midR);

            // The first pixel column for the current row.
            int cqStart = std::max(static_cast<int>(std::floor(0.5f+std::min(cmb, ctb))), 0);
            // The last pixel column for the current row.
            int cqEnd = std::min(static_cast<int>(std::floor(0.5f+std::max(cmb, ctb))), width);

            for(int cq = cqStart; cq < cqEnd; ++cq)
            {
                // Get barycentric coordinates for the current point.
                getBarycentricCoordinates(v1, v2, v3, static_cast<float>(cq)+0.5f, static_cast<float>(rq)+0.5f, l1, l2, l3);

                if(0.f > l1 || 0.f > l2 || 0.f > l3)
                {
                    //continue;
                }

                // The z value for the point.
                float z = v1z*l1+v2z*l2+v3z*l3;

                // Check depth
                float depthValue = depth_.at<float>(rq, cq);
                if(z < depthValue)
                {
                    // Current is behind another, don't draw.
                    continue;
                }

                // The uv values of the point.
                float u, v;
                u = v1u*l1+v2u*l2+v3u*l3;
                v = v1v*l1+v2v*l2+v3v*l3;

                renderPixel<T>(rq, cq, u*fCols, (1.0f-v)*fRows, texture);

                // Update depth buffer.
                depth_.at<float>(rq, cq) = z;
            }
        }
    }
}

template <typename T>
void OdmOrthoPhoto::renderPixel(int row, int col, float s, float t, const cv::Mat &texture)
{
    // The offset of the texture coordinate from its pixel positions.
    float leftF, topF;
    // The position of the top left pixel.
    int left, top;
    // The distance to the left and right pixel from the texture coordinate.
    float dl, dt;
    // The distance to the top and bottom pixel from the texture coordinate.
    float dr, db;
    
    dl = modff(s, &leftF);
    dr = 1.0f - dl;
    dt = modff(t, &topF);
    db = 1.0f - dt;
    
    left = static_cast<int>(leftF);
    top = static_cast<int>(topF);
    
    // The interpolated color values.
    size_t idx = static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(col);
    T *data = reinterpret_cast<T *>(texture.data); // Faster access
    int numChannels = texture.channels();

    for (int i = 0; i < numChannels; i++){
        float value = 0.0f;

        T tl = data[(top) * texture.cols * numChannels + (left) * numChannels + i];
        T tr = data[(top) * texture.cols * numChannels + (left + 1) * numChannels + i];
        T bl = data[(top + 1) * texture.cols * numChannels + (left) * numChannels + i];
        T br = data[(top + 1) * texture.cols * numChannels + (left + 1) * numChannels + i];

        value += static_cast<float>(tl) * dr * db;
        value += static_cast<float>(tr) * dl * db;
        value += static_cast<float>(bl) * dr * dt;
        value += static_cast<float>(br) * dl * dt;

        static_cast<T *>(bands[currentBandIndex + i])[idx] = static_cast<T>(value);
    }

    // Increment the alpha band if the pixel was visible for this band
    // the final alpha band will be set to 255 if alpha == num bands
    // (all bands have information at this pixel)
    static_cast<T *>(alphaBand)[idx] += static_cast<T>(numChannels);
}

void OdmOrthoPhoto::getBarycentricCoordinates(pcl::PointXYZ v1, pcl::PointXYZ v2, pcl::PointXYZ v3, float x, float y, float &l1, float &l2, float &l3) const
{
    // Diff along y.
    float y2y3 = v2.y-v3.y;
    float y1y3 = v1.y-v3.y;
    float y3y1 = v3.y-v1.y;
    float yy3  =  y  -v3.y;
    
    // Diff along x.
    float x3x2 = v3.x-v2.x;
    float x1x3 = v1.x-v3.x;
    float xx3  =  x  -v3.x;
    
    // Normalization factor.
    float norm = (y2y3*x1x3 + x3x2*y1y3);
    
    l1 = (y2y3*(xx3) + x3x2*(yy3)) / norm;
    l2 = (y3y1*(xx3) + x1x3*(yy3)) / norm;
    l3 = 1 - l1 - l2;
}

bool OdmOrthoPhoto::isSliverPolygon(pcl::PointXYZ v1, pcl::PointXYZ v2, pcl::PointXYZ v3) const
{
    // Calculations are made using doubles, to minize rounding errors.
    Eigen::Vector3d a = Eigen::Vector3d(static_cast<double>(v1.x), static_cast<double>(v1.y), static_cast<double>(v1.z));
    Eigen::Vector3d b = Eigen::Vector3d(static_cast<double>(v2.x), static_cast<double>(v2.y), static_cast<double>(v2.z));
    Eigen::Vector3d c = Eigen::Vector3d(static_cast<double>(v3.x), static_cast<double>(v3.y), static_cast<double>(v3.z));
    Eigen::Vector3d dummyVec = (a-b).cross(c-b);

    // Area smaller than, or equal to, floating-point epsilon.
    return std::numeric_limits<float>::epsilon() >= static_cast<float>(std::sqrt(dummyVec.dot(dummyVec))/2.0);
}
*/

// Totally not compatible with all OBJ files, just a subset of those
// that we expect as output from ODM
void OdmOrthoPhoto::loadObjFile(std::string inputFile, TextureMesh &mesh)
{
    std::ifstream fin;
    fs::path p(inputFile);

    fin.open (inputFile.c_str (), std::ios::binary);
    if (!fin.is_open()) throw OdmOrthoPhotoException("Problem reading mesh file: " + inputFile);

    std::string line;
    std::stringstream ss;

    std::string currentFaceMat = "";

    while(std::getline(fin, line)){
        size_t mtllibPos = line.find("mtllib ");
        if (mtllibPos == 0){
            std::string mtlFilesLine = line.substr(std::string("mtllib ").length(), std::string::npos);
            trim(mtlFilesLine);

            auto mtlFiles = split(mtlFilesLine, " ");
            for (auto &mtlFile : mtlFiles){
                trim(mtlFile);
                fs::path mtlRelPath = p.parent_path() / mtlFile;

                if (fs::exists(mtlRelPath)) {
                    // Parse MTL
                    std::string mtlLine;
                    std::ifstream mtlFin(mtlRelPath.string());
                    if (!mtlFin.is_open()) throw OdmOrthoPhotoException("Problem reading MTL file: " + mtlFile);

                    std::string currentMaterial = "";

                    while(std::getline(mtlFin, mtlLine)){
                        if (mtlLine.find("newmtl ") == 0){
                            auto tokens = split(mtlLine, " ");
                            if (tokens.size() >= 2){
                                currentMaterial = tokens[1];
                                log_ << "Found " << currentMaterial << "\n";
                            }
                        }else if (mtlLine.find("map_Kd ") == 0){
                            auto tokens = split(mtlLine, " ");
                            if (tokens.size() >= 2){
                                auto mapFname = tokens[1];
                                if (mapFname.rfind(".") != std::string::npos){
                                    fs::path matPath = p.parent_path() / mapFname;

                                    // Read file in memory
                                    log_ << "Loading " << mapFname << "\n";

                                    cv::Mat texture = cv::imread(matPath.string(), cv::IMREAD_ANYDEPTH | cv::IMREAD_UNCHANGED);

                                    // BGR to RGB when necessary
                                    if (texture.channels() == 3){
                                        cv::cvtColor(texture, texture, cv::COLOR_BGR2RGB);
                                    }

                                    mesh.materials[currentMaterial] = texture;
                                }
                            }
                        }
                    }

                    mtlFin.close();
                }
            }
        }else if (line.find("v ") == 0){
            ss.str(line);
            ss.seekg(2);
            PointXYZ v;
            ss >> v[0];
            ss >> v[1];
            ss >> v[2];
            mesh.vertices.push_back(v);
        }else if (line.find("vt ") == 0){
            ss.str(line);
            ss.seekg(3);
            Tex2D uv;
            ss >> uv[0];
            ss >> uv[1];
            mesh.uvs.push_back(uv);
        }else if (line.find("f ") == 0){
            auto tokens = split(line, " ");
            if (tokens.size() >= 4){
                auto parts = split(tokens[1], "/");
                if (parts.size() >= 2){
                    int av = std::stoi(parts[0]);
                    int at = std::stoi(parts[1]);

                    parts = split(tokens[2], "/");
                    if (parts.size() >= 2){
                        int bv = std::stoi(parts[0]);
                        int bt = std::stoi(parts[1]);

                        parts = split(tokens[3], "/");
                        if (parts.size() >= 2){
                            int cv = std::stoi(parts[0]);
                            int ct = std::stoi(parts[1]);

                            Face f;
                            f.v1 = mesh.vertices[av - 1];
                            f.v2 = mesh.vertices[bv - 1];
                            f.v3 = mesh.vertices[cv - 1];
                            f.t1 = mesh.uvs[at - 1];
                            f.t2 = mesh.uvs[bt - 1];
                            f.t3 = mesh.uvs[ct - 1];

                            mesh.faces[currentFaceMat].push_back(f);
                        }
                    }

                }
            }
        }else if (line.find("usemtl ") == 0){
            auto tokens = split(line, " ");
            if (tokens.size() >= 2){
                currentFaceMat = tokens[1];
                trim(currentFaceMat);
            }
        }
    }

    fin.close();
}
