interface ISiteMetadataResult {
  siteTitle: string;
  siteUrl: string;
  description: string;
  logo: string;
  navLinks: {
    name: string;
    url: string;
  }[];
}

const data: ISiteMetadataResult = {
  siteTitle: 'ChenSoul 运动',
  siteUrl: 'https://run.chensoul.cc',
  logo: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQTtc69JxHNcmN1ETpMUX4dozAgAN6iPjWalQ&usqp=CAU',
  description: 'ChenSoul 运动：跑步、骑行、健走',
  navLinks: [
    {
      name: 'Blog',
      url: 'https://blog.chensoul.cc',
    },
    {
      name: 'Github',
      url: 'https://github.com/chensoul',
    }
  ],
};

export default data;
