#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
//#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

int main()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile("bunny/reconstruction/bun_zipper.ply", *cloud);
	pcl::visualization::PCLVisualizer viewer;
	viewer.addPointCloud(cloud);
	viewer.spin();
}
