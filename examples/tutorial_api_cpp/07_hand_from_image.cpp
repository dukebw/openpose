// ----------------------------- OpenPose C++ API Tutorial - Example 7 - Face from Image -----------------------------
// It reads an image and the hand location, process it, and displays the hand keypoints. In addition,
// it includes all the OpenPose configuration flags.
// Input: An image and the hand rectangle locations.
// Output: OpenPose hand keypoint detection.
// NOTE: This demo is auto-selecting the following flags: `--body 0 --hand --hand_detector 2`

// Command-line user intraface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// Custom OpenPose flags
// Producer
DEFINE_string(image_path, "examples/media/COCO_val2014_000000000241.jpg",
    "Process an image. Read all standard formats (jpg, png, bmp, etc.).");
// Display
DEFINE_bool(no_display,                 false,
    "Enable to disable the visual display.");

/* NOTE(brendan): source: https://stackoverflow.com/a/17820615 */
static std::string
type2str(int type)
{
        std::string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch ( depth ) {
                case CV_8U:  r = "8U"; break;
                case CV_8S:  r = "8S"; break;
                case CV_16U: r = "16U"; break;
                case CV_16S: r = "16S"; break;
                case CV_32S: r = "32S"; break;
                case CV_32F: r = "32F"; break;
                case CV_64F: r = "64F"; break;
                default:     r = "User"; break;
        }

        r += "C";
        r += (chans+'0');

        return r;
}

// This worker will just read and return all the jpg files in a directory
void display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr,
             const cv::Mat& imageToProcess,
             const std::vector<std::array<op::Rectangle<float>, 2>>& hand_rects)
{
    try
    {
        // User's displaying/saving/other processing here
            // datum.cvOutputData: rendered frame with pose or heatmaps
            // datum.poseKeypoints: Array<float> with the estimated pose
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
                auto& handHeatMaps = datumsPtr->at(0)->handHeatMaps;
                const auto num_people = handHeatMaps[0].getSize(0);
                const auto num_joints = handHeatMaps[0].getSize(1);
                const auto height = handHeatMaps[0].getSize(2);
                const auto width = handHeatMaps[0].getSize(3);
                int32_t desired_joint = 0;
                int32_t chanidx = desired_joint % num_joints*height*width;
                const cv::Mat desiredChannelHeatMap{height,
                                                    width,
                                                    CV_32F,
                                                    &handHeatMaps[0].getPtr()[chanidx]};

                /* NOTE(brendan): from 08_heatmaps_from_image.cpp. */
                auto& inputNetData = datumsPtr->at(0)->inputNetData[0];
                op::log("Input net data size: [" + std::to_string(inputNetData.getSize(0)) + ", "
                        + std::to_string(inputNetData.getSize(1)) + ", "
                        + std::to_string(inputNetData.getSize(2)) + ", "
                        + std::to_string(inputNetData.getSize(3)) + "]");
                op::log("Image to process data size: [" + std::to_string(imageToProcess.rows) + ", "
                        + std::to_string(imageToProcess.cols) + "]"
                        + ", type: " + type2str(imageToProcess.type()));
                const cv::Mat inputNetDataB{height,
                                            width,
                                            CV_32F,
                                            &inputNetData.getPtr()[0]};
                const cv::Mat inputNetDataG{height,
                                            width,
                                            CV_32F,
                                            &inputNetData.getPtr()[height*width]};
                const cv::Mat inputNetDataR{height,
                                            width,
                                            CV_32F,
                                            &inputNetData.getPtr()[2*height*width]};
                cv::Mat netInputImage;
                cv::merge(std::vector<cv::Mat>{inputNetDataB, inputNetDataG, inputNetDataR}, netInputImage);
                netInputImage = (netInputImage+0.5)*255;
                cv::Mat netInputImageUint8;
                netInputImage.convertTo(netInputImageUint8, CV_8UC1);

                /* cv::Mat desiredChannelHeatMapUint8; */
                /* desiredChannelHeatMap.convertTo(desiredChannelHeatMapUint8, CV_8UC1); */
                /* // Combining both images */
                /* cv::Mat imageToRender; */
                /* cv::applyColorMap(desiredChannelHeatMapUint8, desiredChannelHeatMapUint8, cv::COLORMAP_JET); */
                /* cv::addWeighted(netInputImageUint8, 0.5, desiredChannelHeatMapUint8, 0.5, 0., imageToRender); */
                /* cv::imwrite("./test.jpg", imageToRender); */

            // Display image
            op::log("Left hand heatmaps size: [" + std::to_string(handHeatMaps[0].getSize(0)) + ", "
                    + std::to_string(handHeatMaps[0].getSize(1)) + ", "
                    + std::to_string(handHeatMaps[0].getSize(2)) + ", "
                    + std::to_string(handHeatMaps[0].getSize(3)) + "]");
            op::log("Right hand heatmaps size: [" + std::to_string(handHeatMaps[0].getSize(0)) + ", "
                    + std::to_string(handHeatMaps[1].getSize(1)) + ", "
                    + std::to_string(handHeatMaps[1].getSize(2)) + ", "
                    + std::to_string(handHeatMaps[1].getSize(3)) + "]");
        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        // Example: How to use the pose keypoints
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            op::log("Body keypoints: " + datumsPtr->at(0)->poseKeypoints.toString(), op::Priority::High);
            op::log("Face keypoints: " + datumsPtr->at(0)->faceKeypoints.toString(), op::Priority::High);
            op::log("Left hand keypoints: " + datumsPtr->at(0)->handKeypoints[0].toString(), op::Priority::High);
            op::log("Right hand keypoints: " + datumsPtr->at(0)->handKeypoints[1].toString(), op::Priority::High);
        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void configureWrapper(op::Wrapper& opWrapper)
{
    try
    {
        // Configuring OpenPose

        // logging_level
        op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
                  __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(FLAGS_face_net_resolution, "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(FLAGS_hand_net_resolution, "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::log("Flag `write_keypoint` is deprecated and will eventually be removed."
                    " Please, use `write_json` instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
            FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
            poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
            FLAGS_part_to_show, FLAGS_model_folder, heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
            (float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
            FLAGS_prototxt_path, FLAGS_caffemodel_path, (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
            op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
        opWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
        opWrapper.configure(wrapperStructExtra);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, FLAGS_write_keypoint, op::stringToDataFormat(FLAGS_write_keypoint_format),
            FLAGS_write_json, FLAGS_write_coco_json, FLAGS_write_coco_json_variants, FLAGS_write_coco_json_variant,
            FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video, FLAGS_write_video_fps,
            FLAGS_write_video_with_audio, FLAGS_write_heatmaps, FLAGS_write_heatmaps_format, FLAGS_write_video_3d,
            FLAGS_write_video_adam, FLAGS_write_bvh, FLAGS_udp_host, FLAGS_udp_port};
        opWrapper.configure(wrapperStructOutput);
        // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

int tutorialApiCpp()
{
    try
    {
        op::log("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // Required flags to enable heatmaps
        FLAGS_body = 0;
        FLAGS_hand = true;
        FLAGS_hand_detector = 2;

        // Configuring OpenPose
        op::log("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        configureWrapper(opWrapper);

        // Starting OpenPose
        op::log("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        // Read image and hand rectangle locations
        const auto imageToProcess = cv::imread(FLAGS_image_path);
        const std::vector<std::array<op::Rectangle<float>, 2>> handRectangles{
            // Left/Right hands of person 0
            std::array<op::Rectangle<float>, 2>{
                op::Rectangle<float>{320.035889f, 377.675049f, 69.300949f, 69.300949f}, // Left hand
                op::Rectangle<float>{0.f, 0.f, 0.f, 0.f}},                              // Right hand
            // Left/Right hands of person 1
            std::array<op::Rectangle<float>, 2>{
                op::Rectangle<float>{80.155792f, 407.673492f, 80.812706f, 80.812706f},  // Left hand
                op::Rectangle<float>{46.449715f, 404.559753f, 98.898178f, 98.898178f}}, // Right hand
            // Left/Right hands of person 2
            std::array<op::Rectangle<float>, 2>{
                op::Rectangle<float>{185.692673f, 303.112244f, 157.587555f, 157.587555f},// Left hand
                op::Rectangle<float>{88.984360f, 268.866547f, 117.818230f, 117.818230f}} // Right hand
        };

        // Create new datum
        auto datumsPtr = std::make_shared<std::vector<std::shared_ptr<op::Datum>>>();
        datumsPtr->emplace_back();
        auto& datumPtr = datumsPtr->at(0);
        datumPtr = std::make_shared<op::Datum>();
        // Fill datum with image and handRectangles
        datumPtr->cvInputData = imageToProcess;
        datumPtr->handRectangles = handRectangles;

        // Process and display image
        opWrapper.emplaceAndPop(datumsPtr);
        if (datumsPtr != nullptr)
        {
            printKeypoints(datumsPtr);
            if (!FLAGS_no_display)
                display(datumsPtr, imageToProcess, handRectangles);
        }
        else
            op::log("Image could not be processed.", op::Priority::High);

        // Info
        op::log("NOTE: In addition with the user flags, this demo has auto-selected the following flags:\n"
                "\t`--body 0 --hand --hand_detector 2`", op::Priority::High);

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return
        return 0;
    }
    catch (const std::exception& e)
    {
        return -1;
    }
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp
    return tutorialApiCpp();
}
