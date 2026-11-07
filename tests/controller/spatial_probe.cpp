// CPU mirror of the CUDA feature definition; input rows are candidate H only.
#include <iomanip>
#include <iostream>
#include "harness/image_loader.hpp"
#include "controller/spatial.hpp"
int main(int argc,char** argv) {
    if(argc!=3 && argc!=4)return 2;
    const unsigned groups=argc==4?std::stoul(argv[3]):3;
    try {
        auto a=stitch::harness::load_ppm(argv[1]),b=stitch::harness::load_ppm(argv[2]);
        stitch::ImageView av{a.pixels.data(),a.width,a.height,a.width*3,stitch::PixelFormat::RGB8};
        stitch::ImageView bv{b.pixels.data(),b.width,b.height,b.width*3,stitch::PixelFormat::RGB8};
        std::cout<<std::setprecision(17);double h[9];
        while(std::cin>>h[0]) {
            for(int i=1;i<9;++i)if(!(std::cin>>h[i]))return 2;
            double out[16];const bool valid=stitch::controller::spatial_observations(av,bv,h,out,groups);
            std::cout<<valid;for(double v:out)std::cout<<','<<v;std::cout<<'\n';
        }
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
