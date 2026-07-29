import setuptools


setuptools.setup(
    name="deeplink_megatron",
    version="0.1.0",
    description="deeplink_megatron pipeline scheduler extension for Megatron-LM",
    packages=setuptools.find_namespace_packages(where="..", include=["deeplink_megatron", "deeplink_megatron.*"]),
    package_dir={"": ".."},
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
