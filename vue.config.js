const path = require('path')
const resolve = dir => path.join(__dirname, dir)
const BundleAnalyzerPlugin = require("webpack-bundle-analyzer").BundleAnalyzerPlugin;
module.exports = {
    publicPath :  './' ,
    runtimeCompiler: true,
    productionSourceMap: false,
    chainWebpack: config => {
        config
      .plugin('html')
      .tap(args => {
        args[0].title= 'AR'
        return args
      })
      config
      .plugin('webpack-bundle-analyzer')
      .use(BundleAnalyzerPlugin)
        config.resolve.alias
            .set('@', resolve('src'))
            
    }


}
