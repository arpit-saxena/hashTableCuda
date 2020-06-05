#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>
#include <chrono>

int main()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud2(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile("bunny/reconstruction/bun_zipper.ply", *cloud);
	pcl::io::loadPLYFile("bunny/reconstruction/bun_zipper.ply", *cloud2);
	pcl::visualization::PCLVisualizer viewer;
	viewer.addPointCloud(cloud, "cloud");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud2, 0, 255, 0);
	viewer.addPointCloud(cloud2, single_color, "cloud2");
	int times = 200, i = 0;
	while (!viewer.wasStopped()) {
		for (auto it = cloud->begin(); it != cloud->end(); ++it) {
			it->x += ((i / times) % 2 == 0) ? 0.001 : -0.001;
		}
		i++;
		viewer.updatePointCloud(cloud, "cloud");
		viewer.spinOnce();
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}
}
