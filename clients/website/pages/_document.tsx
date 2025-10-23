import Document, { Html, Head, Main, NextScript } from 'next/document'

export default class MyDocument extends Document {
  render() {
    const suppressExtensions = `
      (function() {
        try {
          window.addEventListener('error', function(e) {
            var src = (e && e.filename) || '';
            var msg = (e && e.message) || '';
            if (src.indexOf('chrome-extension') > -1 || src.indexOf('moz-extension') > -1 || msg.indexOf('MetaMask') > -1) {
              e.preventDefault && e.preventDefault();
              e.stopPropagation && e.stopPropagation();
              return false;
            }
          }, true);
          window.addEventListener('unhandledrejection', function(e) {
            try {
              var reason = e && e.reason ? (e.reason.message || String(e.reason)) : '';
              if (String(reason).indexOf('MetaMask') > -1 || String(reason).indexOf('chrome-extension') > -1) {
                e.preventDefault && e.preventDefault();
                return false;
              }
            } catch(_){}
          });
        } catch(_){}
      })();
    `
    return (
      <Html lang="en">
        <Head>
          <script dangerouslySetInnerHTML={{ __html: suppressExtensions }} />
        </Head>
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    )
  }
}


