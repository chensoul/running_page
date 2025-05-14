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

const getBasePath = () => {
  const baseUrl = import.meta.env.BASE_URL;
  return baseUrl === '/' ? '' : baseUrl;
};

const data: ISiteMetadataResult = {
  siteTitle: 'Chensoul Run',
  siteUrl: 'https://run.chensoul.cc',
  logo: 'https://blog.chensoul.cc/images/favicon.webp',
  description: 'Personal site and blog',
  navLinks: [
    {
      name: 'Summary',
      url: `${getBasePath()}/summary`,
    },
    {
      name: 'Blog',
      url: 'https://blog.chensoul.cc',
    },
    {
      name: 'About',
      url: 'https://github.com/chensoul',
    },
  ],
};

export default data;