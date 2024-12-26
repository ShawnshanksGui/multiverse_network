#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

namespace nb = nanobind;

namespace madEscape {

// This file creates the python bindings used by the learning code.
// Refer to the nanobind documentation for more details on these functions.
NB_MODULE(madrona_escape_room, m) {
    // Each simulator has a madrona submodule that includes base types
    // like madrona::py::Tensor and madrona::py::PyExecMode.
    madrona::py::setupMadronaSubmodule(m);

    // nb::class_<AJLink>(m, "AJLink")
    //     .def(nb::init<>())  // 默认构造函数
    //     .def_readonly("aj_link", &AJLink::aj_link)  // 只读属性
    //     .def_readonly("link_num", &AJLink::link_num)
    //     .def_readonly("npu_num", &AJLink::npu_num);



    nb::class_<Manager> (m, "SimManager")
        .def("__init__", [](Manager *self,
                            madrona::py::PyExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t rand_seed,
                            bool auto_reset,
                            bool enable_batch_renderer,
                            uint32_t k_aray,
                            uint32_t cc_method
                            // AJLink aj_link)
                            )
        {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .randSeed = (uint32_t)rand_seed,
                .autoReset = auto_reset,
                .enableBatchRenderer = enable_batch_renderer,
                .kAray = (uint32_t)k_aray, // fei add in 20241215
                .ccMethod = (uint32_t)cc_method,
                // .ajLink = aj_link,
            });
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("rand_seed"),
           nb::arg("auto_reset"),
           nb::arg("enable_batch_renderer") = false,
           nb::arg("k_aray") = 4, // fei add in 20241215
           nb::arg("cc_method") = 0
        //    nb::arg("aj_link")
           ) // fei add in 2024120215
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("self_observation_tensor", &Manager::selfObservationTensor)
        .def("partner_observations_tensor", &Manager::partnerObservationsTensor)
        .def("room_entity_observations_tensor",
             &Manager::roomEntityObservationsTensor)
        .def("door_observation_tensor",
             &Manager::doorObservationTensor)
        .def("lidar_tensor", &Manager::lidarTensor)
        .def("steps_remaining_tensor", &Manager::stepsRemainingTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("depth_tensor", &Manager::depthTensor)
        .def("results_tensor", &Manager::resultsTensor)  // fei add in 202412015
        .def("results2_tensor", &Manager::results2Tensor)
        .def("madronaEvents_tensor", &Manager::madronaEventsTensor)
        .def("madronaEventsResult_tensor", &Manager::madronaEventsResultTensor)
        .def("simulation_time_tensor", &Manager::simulationTimeTensor)
        .def("processParams_tensor",&Manager::processParamsTensor)
    ;
}

}





// #include "mgr.hpp"
// #include <madrona/macros.hpp>
// #include <madrona/py/bindings.hpp>
// #include <vector> // 如果使用 std::vector

// namespace nb = nanobind;

// namespace madEscape {

// NB_MODULE(madrona_escape_room, m) {
//     madrona::py::setupMadronaSubmodule(m);

//     nb::class_<AJLink>(m, "AJLink")
//         .def(nb::init<>())
//         .def("get_aj_link", [](const AJLink &self) {
//             return nb::array(nb::dtype<uint32_t>(), {100000, 5}, self.get_aj_link()); // 返回 numpy 数组
//         })
//         .def("get_link_num", &AJLink::link_num)
//         .def("get_npu_num", &AJLink::npu_num);

//     nb::class_<Manager>(m, "SimManager")
//         .def("__init__", [](Manager *self,
//                             madrona::py::PyExecMode exec_mode,
//                             int64_t gpu_id,
//                             int64_t num_worlds,
//                             int64_t rand_seed,
//                             bool auto_reset,
//                             bool enable_batch_renderer,
//                             uint32_t k_aray,
//                             uint32_t cc_method,
//                             AJLink aj_link)  
//         {
//             new (self) Manager(Manager::Config {
//                 .execMode = exec_mode,
//                 .gpuID = (int)gpu_id,
//                 .numWorlds = (uint32_t)num_worlds,
//                 .randSeed = (uint32_t)rand_seed,
//                 .autoReset = auto_reset,
//                 .enableBatchRenderer = enable_batch_renderer,
//                 .kAray = (uint32_t)k_aray,
//                 .ccMethod = (uint32_t)cc_method,
//                 .ajLink = aj_link,
//             });
//         }, nb::arg("exec_mode"),
//            nb::arg("gpu_id"),
//            nb::arg("num_worlds"),
//            nb::arg("rand_seed"),
//            nb::arg("auto_reset"),
//            nb::arg("enable_batch_renderer") = false,
//            nb::arg("k_aray") = 4,
//            nb::arg("cc_method") = 0,
//            nb::arg("aj_link")
//         )
//         .def("step", &Manager::step)
//         .def("reset_tensor", &Manager::resetTensor)
//         .def("action_tensor", &Manager::actionTensor)
//         .def("reward_tensor", &Manager::rewardTensor)
//         .def("done_tensor", &Manager::doneTensor)
//         .def("self_observation_tensor", &Manager::selfObservationTensor)
//         .def("partner_observations_tensor", &Manager::partnerObservationsTensor)
//         .def("room_entity_observations_tensor", &Manager::roomEntityObservationsTensor)
//         .def("door_observation_tensor", &Manager::doorObservationTensor)
//         .def("lidar_tensor", &Manager::lidarTensor)
//         .def("steps_remaining_tensor", &Manager::stepsRemainingTensor)
//         .def("rgb_tensor", &Manager::rgbTensor)
//         .def("depth_tensor", &Manager::depthTensor)
//         .def("results_tensor", &Manager::resultsTensor)
//         .def("results2_tensor", &Manager::results2Tensor)
//         .def("madronaEvents_tensor", &Manager::madronaEventsTensor)
//         .def("madronaEventsResult_tensor", &Manager::madronaEventsResultTensor)
//         .def("simulation_time_tensor", &Manager::simulationTimeTensor)
//         .def("processParams_tensor", &Manager::processParamsTensor);
// }

// }
