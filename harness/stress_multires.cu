// Native CH-05 driver. S6-only accepts H_gt; end-to-end cannot accept any H input.
#include "harness/stress_io.hpp"
#include "harness/pipeline_runner.hpp"
#include <chrono>
#include <cmath>
#include <map>
#include <climits>

using namespace stitch;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using namespace stitch::stress;

struct Options {
    std::string track, mode, a, b, h, rois, output;
    int width=0, height=0, canvas_w=0, canvas_h=0, warmups=2, repeats=10, expected_sm=75;
    std::size_t host_budget=512u*1024u*1024u, reserve=256u*1024u*1024u;
    bool full=false;
    harness::HarnessConfig config{};
};
Options parse(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i=1; i<argc; ++i) {
        const std::string key=argv[i];
        if (key=="--export-full") { if (!args.emplace(key,"1").second) throw std::invalid_argument("duplicate flag"); continue; }
        if (i+1>=argc || !args.emplace(key,argv[++i]).second) throw std::invalid_argument("missing/duplicate argument");
    }
    const std::vector<std::string> known={"--track","--mode","--a","--b","--h-file","--rois","--output","--width","--height",
        "--canvas-width","--canvas-height","--warmups","--repeats","--expected-sm","--host-budget-bytes","--device-reserve-bytes",
        "--scale","--keypoints","--ransac","--seed","--export-full"};
    for (auto& entry:args) if (std::find(known.begin(),known.end(),entry.first)==known.end()) throw std::invalid_argument("unknown argument");
    auto required=[&](const char* name) { if (!args.count(name)) throw std::invalid_argument("required argument missing"); return args.at(name); };
    auto integer=[&](const char* name, std::size_t fallback) {
        if (!args.count(name)) return fallback;
        const auto value=args.at(name);
        if (value.empty() || value.find_first_not_of("0123456789")!=std::string::npos) throw std::invalid_argument("nonnegative integer required");
        return static_cast<std::size_t>(std::stoull(value));
    };
    auto dimension=[&](const char* name) { const auto v=integer(name,0); if (!v || v>static_cast<std::size_t>(INT_MAX)) throw std::invalid_argument("invalid dimension"); return static_cast<int>(v); };
    Options o; o.track=required("--track"); o.mode=required("--mode"); o.a=required("--a"); o.b=required("--b"); o.output=required("--output");
    o.width=dimension("--width"); o.height=dimension("--height");
    if (o.track!="s6_only" && o.track!="end_to_end") throw std::invalid_argument("unknown track");
    if (o.track=="end_to_end" && (args.count("--h-file") || args.count("--rois") || args.count("--export-full")))
        throw std::invalid_argument("end-to-end forbids ground-truth and render-oracle arguments");
    if (o.track=="s6_only") { o.h=required("--h-file"); o.rois=required("--rois"); }
    else { o.canvas_w=dimension("--canvas-width"); o.canvas_h=dimension("--canvas-height"); }
    if (o.mode=="feather") o.config.blend_mode=s6::BlendMode::Feather;
    else if (o.mode=="multiband_3") o.config.blend_mode=s6::BlendMode::Multiband3;
    else if (o.mode=="multiband_5") o.config.blend_mode=s6::BlendMode::Multiband5;
    else throw std::invalid_argument("unknown blender");
    const auto warm=integer("--warmups",2), repeats=integer("--repeats",10);
    if (warm>1000 || repeats<1 || repeats>1000) throw std::invalid_argument("invalid repetitions");
    o.warmups=warm; o.repeats=repeats; o.expected_sm=integer("--expected-sm",75);
    o.host_budget=integer("--host-budget-bytes",o.host_budget); o.reserve=integer("--device-reserve-bytes",o.reserve);
    const auto k=integer("--keypoints",1000), r=integer("--ransac",256);
    if (!k || k>INT_MAX || !r || r>INT_MAX) throw std::invalid_argument("invalid K/R");
    o.config.K=k; o.config.R=r; o.config.seed=integer("--seed",0); o.config.config_id="stress_fixed";
    if (args.count("--scale")) { std::size_t end=0; o.config.S=std::stod(args.at("--scale"),&end);
        if (end!=args.at("--scale").size() || !std::isfinite(o.config.S) || !(o.config.S>0 && o.config.S<=1)) throw std::invalid_argument("invalid scale"); }
    o.config.blend_mode_str=o.mode; o.full=args.count("--export-full");
    return o;
}

void write_row(std::ostream& out, int repeat, bool warmup, const std::string& status,
               const std::array<double,8>& times, const harness::RunRecord* record,
               const s6::Geometry* geometry, std::size_t workspace) {
    auto number=[](double value) { std::ostringstream s; s << std::setprecision(17) << value; return s.str(); };
    std::vector<std::string> fields{std::to_string(repeat), warmup?"1":"0", status};
    for (double value:times) fields.push_back(value>=0 && std::isfinite(value)?number(value):"");
    fields.push_back(record?std::to_string(record->num_inliers):"");
    fields.push_back(record?number(record->inlier_ratio):"");
    for (int i=0;i<9;++i) fields.push_back(record && status=="SUCCESS"?number(record->H[i]):"");
    if (geometry) {
        for (int value:{geometry->width,geometry->height,geometry->origin_x,geometry->origin_y}) fields.push_back(std::to_string(value));
        fields.push_back(std::to_string(workspace));
    } else for (int i=0;i<5;++i) fields.emplace_back();
    for (std::size_t i=0;i<fields.size();++i) { if (i) out << ','; out << fields[i]; }
    out << '\n';
}

int main(int argc,char** argv) {
    Options o;
    try { o=parse(argc,argv); if (fs::exists(o.output)) throw std::runtime_error("output already exists"); fs::create_directories(o.output); }
    catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 2; }
    std::string status="ERR_RUNNER", reason, gpu_name;
    int major=0, minor=0, cw=0, ch=0, ox=0, oy=0;
    int runtime_version=0, driver_version=0;
    std::size_t workspace=0, allocated_workspace=0, device_bytes=0, free_bytes=0, total_bytes=0;
    std::size_t pipeline_workspace_capacity=0;
    bool workspace_queried=false;
    std::ostringstream rows; rows << std::setprecision(17);
    rows << "repeat_idx,is_warmup,status_code,t_s1,t_s2,t_s3,t_s4,t_s5,t_s6,t_gpu_steady,t_host_e2e,inlier_count,inlier_ratio,H00,H01,H02,H10,H11,H12,H20,H21,H22,canvas_width,canvas_height,origin_x,origin_y,workspace_bytes\n";
    try {
        admission::fit(admission::add(admission::multiply(admission::multiply(o.width,o.height),6),64u*1024u*1024u),o.host_budget,0,"ERR_HOST_OOM");
        s6::Geometry geometry{};
        if (o.track=="s6_only") {
            double h[9]; std::ifstream in(o.h);
            for (double& v:h) if (!(in>>v) || !std::isfinite(v)) throw admission::Error("ERR_INPUT");
            in>>std::ws; if (!in.eof()) throw admission::Error("ERR_INPUT");
            const auto result=s6::prepare_geometry(o.width,o.height,o.width,o.height,h,{},&geometry);
            if (result!=Status::OK) throw admission::Error(result==Status::INSUFFICIENT_WORKSPACE?"ERR_CANVAS_OVERFLOW":"ERR_DEGENERATE_GEOM");
            cw=geometry.width; ch=geometry.height; ox=geometry.origin_x; oy=geometry.origin_y;
            if (s6::s6_workspace_bytes(geometry,o.config.blend_mode,&workspace)!=Status::OK) throw admission::Error("ERR_CAPACITY");
            workspace_queried=true;
            allocated_workspace=workspace?admission::add(workspace,256):0;
            pipeline_workspace_capacity=workspace;
            device_bytes=admission::add(admission::multiply(2,admission::image_bytes(o.width,o.height,3)),
                admission::add(admission::image_bytes(cw,ch,3),admission::add(admission::image_bytes(cw,ch,1),allocated_workspace)));
            if (o.full) admission::fit(admission::add(admission::multiply(admission::multiply(cw,ch),3),
                admission::multiply(admission::multiply(o.width,o.height),6)),o.host_budget,16u*1024u*1024u,"ERR_HOST_OOM");
        } else {
            const auto plan=harness::PipelineRunner::query_memory(o.width,o.height,o.config.K,o.config.R,o.canvas_w,o.canvas_h,o.config.blend_mode);
            device_bytes=plan.device_bytes; allocated_workspace=plan.s6_workspace_bytes?plan.s6_workspace_bytes+256:0;
            pipeline_workspace_capacity=plan.workspace_bytes;
        }
        int devices=0; cuda(cudaGetDeviceCount(&devices)); if (!devices) throw admission::Error("ERR_NO_DEVICE");
        int device=0; cuda(cudaGetDevice(&device)); cudaDeviceProp props{}; cuda(cudaGetDeviceProperties(&props,device));
        gpu_name=props.name; major=props.major; minor=props.minor;
        cuda(cudaRuntimeGetVersion(&runtime_version)); cuda(cudaDriverGetVersion(&driver_version));
        if (10*major+minor!=o.expected_sm) throw admission::Error("ERR_DEVICE_MISMATCH");
        if ((o.expected_sm==75 && gpu_name!="Tesla T4" && gpu_name!="NVIDIA T4" && gpu_name!="T4") ||
            (o.expected_sm==80 && gpu_name.find("A100")==std::string::npos) ||
            (o.expected_sm==89 && gpu_name.find("L4")==std::string::npos)) throw admission::Error("ERR_DEVICE_MISMATCH");
        cuda(cudaMemGetInfo(&free_bytes,&total_bytes)); admission::fit(device_bytes,free_bytes,o.reserve,"ERR_DEVICE_OOM");
        auto a=image(o.a,o.width,o.height), b=image(o.b,o.width,o.height);
        if (o.track=="s6_only") {
            DeviceImage da(o.width,o.height,3), db(o.width,o.height,3), output(cw,ch,3), mask(cw,ch,1);
            Buffer work(allocated_workspace); Stream stream; Event start,stop;
            if (workspace) cuda(cudaMemset(static_cast<std::uint8_t*>(work.data)+workspace,0xa5,256));
            da.upload(a); db.upload(b);
            for (int i=0;i<o.warmups+o.repeats;++i) {
                const auto before=Clock::now(); cuda(cudaEventRecord(start.value,stream.value));
                const auto rendered=s6::render(da.view,db.view,geometry,o.config.blend_mode,output.view,mask.view,work.data,workspace,stream.value);
                if (rendered!=Status::OK) throw admission::Error(rendered==Status::CUDA_ERROR?"ERR_CUDA":"ERR_CAPACITY");
                cuda(cudaEventRecord(stop.value,stream.value)); cuda(cudaEventSynchronize(stop.value));
                const double host=std::chrono::duration<double,std::milli>(Clock::now()-before).count();
                float gpu=0; cuda(cudaEventElapsedTime(&gpu,start.value,stop.value));
                write_row(rows,i<o.warmups?i:i-o.warmups,i<o.warmups,"SUCCESS",{-1,-1,-1,-1,-1,gpu,gpu,host},nullptr,&geometry,workspace);
            }
            da.padding(); db.padding(); output.padding(); mask.padding();
            if (workspace) { std::array<std::uint8_t,256> guard{}; cuda(cudaMemcpy(guard.data(),static_cast<std::uint8_t*>(work.data)+workspace,256,cudaMemcpyDeviceToHost));
                for (auto v:guard) if (v!=0xa5) throw admission::Error("ERR_CORRECTNESS"); }
            std::ifstream roi_file(o.rois); std::array<int,4> box{}; int index=0;
            while (roi_file>>box[0]>>box[1]>>box[2]>>box[3]) {
                if (++index>32) throw admission::Error("ERR_CAPACITY");
                output.save(fs::path(o.output)/("roi"+std::to_string(index-1)+".rgb"),box);
                mask.save(fs::path(o.output)/("roi"+std::to_string(index-1)+".mask"),box);
            }
            if (!roi_file.eof() || index==0) throw admission::Error("ERR_INPUT");
            if (o.full) { output.save(fs::path(o.output)/"full.rgb",{0,0,cw,ch}); mask.save(fs::path(o.output)/"full.mask",{0,0,cw,ch}); }
        } else {
            harness::PipelineRunner runner;
            runner.initialize(o.width,o.height,o.config.K,o.config.R,o.canvas_w,o.canvas_h,o.config.blend_mode,o.reserve);
            runner.upload_pair(a,b);
            for (int i=0;i<o.warmups+o.repeats;++i) {
                harness::RunRecord r{};
                const auto before=Clock::now();
                runner.execute_run(o.config,"stress","source-hash-in-python","synthetic_scaling","synthetic","single",i<o.warmups?i:i-o.warmups,i<o.warmups,&r);
                runner.synchronize();
                const double host=std::chrono::duration<double,std::milli>(Clock::now()-before).count();
                if (r.status=="OK") {
                    if (s6::prepare_geometry(o.width,o.height,o.width,o.height,r.H,{},&geometry)!=Status::OK) throw admission::Error("ERR_DEGENERATE_GEOM");
                    cw=geometry.width; ch=geometry.height; ox=geometry.origin_x; oy=geometry.origin_y;
                    if (s6::s6_workspace_bytes(geometry,o.config.blend_mode,&workspace)!=Status::OK) throw admission::Error("ERR_CAPACITY");
                    workspace_queried=true;
                }
                write_row(rows,r.repeat_idx,r.is_warmup,r.status=="OK"?"SUCCESS":r.status,
                    {r.status=="OK"?r.t_s1:-1,r.status=="OK"?r.t_s2:-1,r.status=="OK"?r.t_s3:-1,
                     r.status=="OK"?r.t_s4:-1,r.status=="OK"?r.t_s5:-1,r.status=="OK"?r.t_s6:-1,
                     runner.last_fixed_gpu_span_ms(),host},&r,r.status=="OK"?&geometry:nullptr,workspace);
            }
        }
        status="COMPLETED";
    } catch (const admission::Error& e) { status=e.code; reason=e.what(); }
      catch (const harness::CudaError& e) { status=e.code==cudaErrorMemoryAllocation?"ERR_DEVICE_OOM":"ERR_CUDA"; reason=e.what(); }
      catch (const std::bad_alloc& e) { status="ERR_HOST_OOM"; reason=e.what(); }
      catch (const std::exception& e) { status="ERR_RUNNER"; reason=e.what(); }
    try {
        publish(fs::path(o.output)/"native_metrics.csv",rows.str());
        std::ostringstream meta; meta << std::setprecision(17)
            << "{\"schema_version\":\"multires-native-v1\",\"status_code\":" << quote(status) << ",\"reason\":" << quote(reason)
            << ",\"track\":" << quote(o.track) << ",\"blend_mode\":" << quote(o.mode)
            << ",\"gpu_name\":" << quote(gpu_name) << ",\"sm\":" << major*10+minor
            << ",\"cuda_runtime_version\":" << runtime_version << ",\"cuda_driver_version\":" << driver_version
            << ",\"workspace_bytes\":" << workspace << ",\"workspace_reserved_bytes\":" << allocated_workspace
            << ",\"workspace_query_status\":" << quote(workspace_queried?"QUERIED_ACTUAL_GEOMETRY":"NOT_QUERIED_ACTUAL_GEOMETRY")
            << ",\"workspace_query_scope\":\"s6_actual_geometry\",\"pipeline_workspace_capacity_bytes\":" << pipeline_workspace_capacity
            << ",\"workspace_allocated_bytes\":" << (status=="COMPLETED"?std::to_string(allocated_workspace):"null")
            << ",\"declared_device_bytes\":" << device_bytes << ",\"free_device_bytes_before\":" << free_bytes
            << ",\"total_device_bytes\":" << total_bytes << ",\"peak_vram_bytes\":null"
            << ",\"canvas_width\":" << cw << ",\"canvas_height\":" << ch << ",\"origin_x\":" << ox << ",\"origin_y\":" << oy
            << ",\"homography_source\":" << quote(o.track=="s6_only"?"frozen_ground_truth":"s5_estimated_original_rgb")
            << ",\"input_width\":" << o.width << ",\"input_height\":" << o.height
            << ",\"seed\":" << o.config.seed << ",\"warmups\":" << o.warmups << ",\"repeats\":" << o.repeats
            << ",\"S\":" << o.config.S << ",\"K\":" << o.config.K << ",\"R\":" << o.config.R
            << ",\"memory_scope\":\"declared_allocations_not_peak_vram\"}";
        publish(fs::path(o.output)/"native_receipt.json",meta.str());
    } catch (const std::exception& e) { std::fprintf(stderr,"artifact publication failed: %s\n",e.what()); return 2; }
    return 0;
}
