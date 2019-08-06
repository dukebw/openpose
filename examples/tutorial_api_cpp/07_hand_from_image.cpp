// ----------------------------- OpenPose C++ API Tutorial - Example 7 - Face from Image -----------------------------
// It reads an image and the hand location, process it, and displays the hand keypoints. In addition,
// it includes all the OpenPose configuration flags.
// Input: An image and the hand rectangle locations.
// Output: OpenPose hand keypoint detection.
// NOTE: This demo is auto-selecting the following flags: `--body 0 --hand --hand_detector 2`

// Command-line user intraface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include "json.hpp"
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>
#include <boost/filesystem.hpp>
#include <assert.h>
#include <glob.h>

namespace boostfs = boost::filesystem;

// Custom OpenPose flags
// Producer
DEFINE_string(img_dir, "./",
    "Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).");
DEFINE_string(proposal_dir, "./", "Read bbox proposals.");
DEFINE_string(output_dir, "./", "Output directory.");
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

/**
 * NOTE(brendan): based on
 * https://github.com/dukebw/openpose/blob/f09e48820a44480d4c1197b8a2228ce76c073c68/src/openpose/hand/handExtractorCaffe.cpp#L44.
 */
static void
cropFrame(cv::Mat& handImage,
          const cv::Mat& cvInputData,
          const op::Rectangle<float>& handRectangle,
          const int netInputSide,
          const op::Point<int>& netOutputSize,
          const bool mirrorImage)
{
        try
        {
                // Resize image to hands positions
                const auto scaleLeftHand = handRectangle.width / (float)netInputSide;
                cv::Mat affineMatrix = cv::Mat::eye(2,3,CV_64F);
                if (mirrorImage)
                        affineMatrix.at<double>(0,0) = -scaleLeftHand;
                else
                        affineMatrix.at<double>(0,0) = scaleLeftHand;
                affineMatrix.at<double>(1,1) = scaleLeftHand;
                if (mirrorImage)
                        affineMatrix.at<double>(0,2) = handRectangle.x + handRectangle.width;
                else
                        affineMatrix.at<double>(0,2) = handRectangle.x;
                affineMatrix.at<double>(1,2) = handRectangle.y;
                cv::warpAffine(cvInputData, handImage, affineMatrix, cv::Size{netOutputSize.x, netOutputSize.y},
                               CV_INTER_LINEAR | CV_WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar{0,0,0});
                // CV_INTER_CUBIC | CV_WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar{0,0,0});
                // cv::Mat -> float*
        }
        catch (const std::exception& e)
        {
                op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
}

static void
display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr,
        const cv::Mat& imageToProcess,
        const boostfs::path& outpath)
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
                cv::Mat imageToRender = cv::Mat::zeros(cv::Size{height, width}, CV_8UC3);
                for (int32_t jointidx = 0;
                     jointidx < num_joints;
                     ++jointidx) {
                        cv::Mat desiredChannelHeatMap{
                                height,
                                width,
                                CV_32F,
                                handHeatMaps[0].getPtr() + jointidx*height*width};

                        cv::Mat desiredChannelHeatMapUint8;
                        desiredChannelHeatMap.convertTo(desiredChannelHeatMapUint8,
                                                        CV_8UC1);

                        // Combining both images
                        cv::Mat heat_colormap;
                        cv::applyColorMap(desiredChannelHeatMapUint8,
                                          heat_colormap,
                                          cv::COLORMAP_JET);
                        for (int32_t j = 0;
                             j < height;
                             ++j) {
                                for (int32_t i = 0;
                                     i < width;
                                     ++i) {
                                        if (desiredChannelHeatMapUint8.at<uchar>(j, i) < 8)
                                                continue;
                                        imageToRender.at<cv::Point3_<uchar>>(j, i) = 0.5*(imageToRender.at<cv::Point3_<uchar>>(j, i) +
                                                                                          heat_colormap.at<cv::Point3_<uchar>>(j, i));
                                }
                        }
                }

                cv::addWeighted(imageToRender,
                                0.5,
                                imageToProcess,
                                0.5,
                                0.,
                                imageToRender);
                cv::imwrite(outpath.c_str(), imageToRender);
        }
        else
        {
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
        }
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

        glob_t globbuf;
        int ret = glob((FLAGS_proposal_dir + "/*.json").c_str(),
                       GLOB_TILDE,
                       NULL,
                       &globbuf);
        assert(ret == 0);

        // Read image and hand rectangle locations
        printf("gl_pathc: %lu\n", globbuf.gl_pathc);
        for (size_t pathidx = 0;
             pathidx < globbuf.gl_pathc;
             ++pathidx) {
                boostfs::path proppath{globbuf.gl_pathv[pathidx]};
                boostfs::path impath{FLAGS_img_dir};
                impath += boostfs::path{proppath}.replace_extension("jpg").filename();
                boostfs::path outpath{FLAGS_output_dir};
                outpath += boostfs::path{proppath}.replace_extension("").filename();
                outpath += boostfs::path{"_heatmap.jpg"};
                printf("impath %s proppath %s outpath %s\n",
                       impath.c_str(),
                       proppath.c_str(),
                       outpath.c_str());

                const auto imageToProcess = cv::imread(impath.c_str());

                std::ifstream propifs{proppath.c_str()};
                nlohmann::json props;
                propifs >> props;

                std::vector<std::array<op::Rectangle<float>, 2>> handRectangles;
                for (auto const& prop: props) {
                        assert(prop["bbox"].size() == 4);

                        float bbox_len = op::fastMax(prop["bbox"].at(2), prop["bbox"].at(3));
                        float halflen = bbox_len / 2.0f;
                        bbox_len += halflen;
                        float x0 = op::fastMax(static_cast<float>(prop["bbox"].at(0)) - halflen, 0.f);
                        float y0 = op::fastMax(static_cast<float>(prop["bbox"].at(1)) - halflen, 0.f);
                        std::array<op::Rectangle<float>, 2> next_bbox{
                                op::Rectangle<float>{0.f, 0.f, 0.f, 0.f},
                                op::Rectangle<float>{x0,
                                                     y0,
                                                     bbox_len,
                                                     bbox_len}};
                        handRectangles.push_back(next_bbox);
                }

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
                        datumPtr->netOutputSize.x = 368;
                        datumPtr->netOutputSize.y = 368;
                        const auto netInputSide = op::fastMin(datumPtr->netOutputSize.x,
                                                              datumPtr->netOutputSize.y);
                        auto& inputNetData = datumsPtr->at(0)->inputNetData[0];
                        cv::Mat handImage;
                        cropFrame(handImage,
                                  imageToProcess,
                                  handRectangles.at(0).at(1),
                                  netInputSide,
                                  datumPtr->netOutputSize,
                                  false);

                        if (!FLAGS_no_display)
                                display(datumsPtr, handImage, outpath);
                }
                else
                {
                        op::log("Image could not be processed.", op::Priority::High);
                }
                printf("display\n");
        }

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
