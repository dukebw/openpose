// ----------------------------- OpenPose C++ API Tutorial - Example 7 - Face from Image -----------------------------
// It reads an image and the hand location, process it, and displays the hand keypoints. In addition,
// it includes all the OpenPose configuration flags.
// Input: An image and the hand rectangle locations.
// Output: OpenPose hand keypoint detection.
// NOTE: This demo is auto-selecting the following flags: `--body 0 --hand --hand_detector 2`

// Command-line user intraface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include "openpose/flags.hpp"
// OpenPose dependencies
#include "openpose/headers.hpp"
#include "cnpy.h"
#include "json.hpp"
#include "boost/filesystem.hpp"
#include <assert.h>
#include <glob.h>

namespace boostfs = boost::filesystem;

// Custom OpenPose flags
// Producer
DEFINE_string(img_dir,
              "./",
              "Process a directory of images. Read all standard formats (jpg, "
              "png, bmp, etc.).");
DEFINE_string(proposal_dir, "./", "Read bbox proposals.");
DEFINE_string(sequence_relative_dir,
              "./",
              "Relative sequence directory, e.g., "
              "P05/P05_06/P05_06_10592_pour-sugar-in-cup.");
DEFINE_string(output_dir, "./", "Output directory.");
// Display
DEFINE_bool(no_display, false, "Enable to disable the visual display.");

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

static op::Rectangle<float>
get_rect(const nlohmann::json& bbox)
{
        float bbox_len = op::fastMax(bbox.at(2), bbox.at(3));
        float halflen = bbox_len/2.f;
        bbox_len += halflen;
        float x0 = op::fastMax(static_cast<float>(bbox.at(0)) - halflen, 0.f);
        float y0 = op::fastMax(static_cast<float>(bbox.at(1)) - halflen, 0.f);

        return op::Rectangle<float>{x0, y0, bbox_len, bbox_len};
}

static boostfs::path
addsuffix(const boostfs::path& outpath, const std::string& suffix)
{
        boostfs::path newpath{outpath};
        newpath += boostfs::path{suffix};

        return newpath;
}

static void
encode_heatmaps(std::vector<uchar>& heatmapsbin,
                std::vector<int64_t>& heatmapsbin_sizes,
                op::Array<float>& hand_heatmap,
                const std::vector<int>& compression_params)
{
        const size_t num_joints = hand_heatmap.getSize(1);
        auto height = hand_heatmap.getSize(2);
        auto width = hand_heatmap.getSize(3);
        for (int32_t jointidx = 0;
             jointidx < num_joints;
             ++jointidx) {
                cv::Mat joint_heatmap{
                        height,
                        width,
                        CV_32F,
                        (hand_heatmap.getPtr() + jointidx*height*width)};

                cv::Mat joint_hmap_small;
                /**
                 * NOTE(brendan): reduced side length by factor of 4 for
                 * storage
                 */
                cv::resize(joint_heatmap,
                           joint_hmap_small,
                           cv::Size{92, 92},
                           0,
                           0,
                           cv::INTER_LINEAR);

                cv::Mat joint_heatmap_u8;
                joint_hmap_small.convertTo(joint_heatmap_u8, CV_8UC1);

                std::vector<uchar> outbuf;
                bool ret = cv::imencode(".png",
                                        joint_heatmap_u8,
                                        outbuf,
                                        compression_params);
                assert(ret);

                heatmapsbin.insert(heatmapsbin.end(),
                                   outbuf.begin(),
                                   outbuf.end());
                heatmapsbin_sizes.push_back(outbuf.size());
        }
}

/**
 * NOTE(brendan): Take in a subdirectory that is full of images, e.g.,
 * /path/to/starter-kit-action-recognition/data/interim/rgb_train_segments/P05/P05_06/P05_06_10592_pour-sugar-in-cup
 * and emit all the hand pose heatmaps into
 * FLAGS_output_dir/P05/P05_06/P05_06_10592_pour-sugar-in-cup/heatmaps.
 * Also visualize the heatmaps in FLAGS_output_dir/P05/P05_06/P05_06_10592_pour-sugar-in-cup/display,
 * unless FLAGS_no_display is set.
 */
int tutorialApiCpp(void)
{
    try {
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

        boostfs::path propdir{FLAGS_proposal_dir};
        propdir /= FLAGS_sequence_relative_dir;
        printf("%s\n", propdir.c_str());

        glob_t globbuf;
        int ret = glob((propdir/"*.json").c_str(), GLOB_TILDE, NULL, &globbuf);
        assert(ret == 0);

        // Read image and hand rectangle locations
        for (size_t pathidx = 0;
             pathidx < globbuf.gl_pathc;
             ++pathidx) {
                boostfs::path proppath{globbuf.gl_pathv[pathidx]};
                boostfs::path impath{FLAGS_img_dir};
                impath /= FLAGS_sequence_relative_dir;
                impath /= boostfs::path{proppath}.replace_extension("jpg").filename();
                boostfs::path outpath{FLAGS_output_dir};
                outpath /= FLAGS_sequence_relative_dir;
                boostfs::create_directories(outpath);
                outpath /= boostfs::path{proppath}.replace_extension("").filename();
                printf("impath %s proppath %s outpath %s\n",
                       impath.c_str(),
                       proppath.c_str(),
                       outpath.c_str());

                const auto imageToProcess = cv::imread(impath.c_str());

                std::ifstream propifs{proppath.c_str()};
                nlohmann::json props;
                propifs >> props;

                constexpr int LEFT_INDEX = 1;
                constexpr int RIGHT_INDEX = 2;
                auto right = op::Rectangle<float>{0.f, 0.f, 0.f, 0.f};
                float right_score = 0.f;
                auto left = op::Rectangle<float>{0.f, 0.f, 0.f, 0.f};

                float left_score = 0.f;
                uint32_t num_left = 0;
                uint32_t num_right = 0;
                for (size_t i = 0;
                     i < props.size();
                     ++i) {
                        auto& prop = props[i];
                        if (prop["label"] == LEFT_INDEX) {
                                ++num_left;
                        } else {
                                assert(prop["label"] == RIGHT_INDEX);
                                ++num_right;
                        }
                }

                for (size_t i = 0;
                     i < props.size();
                     ++i) {
                        auto& prop = props[i];
                        assert(prop["bbox"].size() == 4);
                        if (prop["label"] == LEFT_INDEX) {
                                if (prop["score"] > left_score) {
                                        /**
                                         * NOTE(brendan): when there are
                                         * multiple right and no left
                                         * handboxes, set the second highest
                                         * scoring box to left, and vice versa.
                                         */
                                        if (num_right == 0) {
                                                right = left;
                                        }
                                        left = get_rect(prop["bbox"]);
                                        left_score = prop["score"];
                                }
                        } else {
                                assert(prop["label"] == RIGHT_INDEX);
                                if (prop["score"] > right_score) {
                                        if (num_left == 0) {
                                                left = right;
                                        }
                                        right = get_rect(prop["bbox"]);
                                        right_score = prop["score"];
                                }
                        }
                }

                std::vector<std::array<op::Rectangle<float>, 2>> handRectangles;
                auto both_hands = std::array<op::Rectangle<float>, 2>{left, right};
                handRectangles.push_back(both_hands);

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
                assert(datumsPtr != nullptr);

                assert(FLAGS_no_display);
                datumPtr->netOutputSize.x = 368;
                datumPtr->netOutputSize.y = 368;
                auto netInputSide = op::fastMin(datumPtr->netOutputSize.x,
                                                datumPtr->netOutputSize.y);
                std::vector<int> compression_params;
                compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
                compression_params.push_back(9);
                std::vector<uchar> heatmapsbin;
                std::vector<int64_t> heatmapsbin_sizes;
                cv::Mat handImage;
                /* TODO(brendan): only do this if there is a proposal. */
                encode_heatmaps(heatmapsbin,
                                heatmapsbin_sizes,
                                datumsPtr->at(0)->handHeatMaps[0],
                                compression_params);

                encode_heatmaps(heatmapsbin,
                                heatmapsbin_sizes,
                                datumsPtr->at(0)->handHeatMaps[1],
                                compression_params);

                auto outpath_heatmaps = addsuffix(outpath, "_heatmaps.npy");
                cnpy::npy_save(outpath_heatmaps.string(), heatmapsbin, "w");
                auto outpath_hmap_sizes = addsuffix(outpath, "_heatmap_sizes.npy");
                cnpy::npy_save(outpath_hmap_sizes.string(), heatmapsbin_sizes, "w");
        }

        // Info
        op::log("NOTE: In addition with the user flags, this demo has auto-selected the following flags:\n"
                "\t`--body 0 --hand --hand_detector 2`", op::Priority::High);

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}

int main(int argc, char **argv)
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp
    return tutorialApiCpp();
}
