/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
(() => {
var exports = {};
exports.id = "pages/_app";
exports.ids = ["pages/_app"];
exports.modules = {

/***/ "./pages/_app.tsx":
/*!************************!*\
  !*** ./pages/_app.tsx ***!
  \************************/
/***/ ((module, __webpack_exports__, __webpack_require__) => {

"use strict";
eval("__webpack_require__.a(module, async (__webpack_handle_async_dependencies__, __webpack_async_result__) => { try {\n__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   \"default\": () => (/* binding */ App)\n/* harmony export */ });\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react/jsx-dev-runtime */ \"react/jsx-dev-runtime\");\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__);\n/* harmony import */ var react_hot_toast__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react-hot-toast */ \"react-hot-toast\");\n/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ \"react\");\n/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);\n/* harmony import */ var _styles_globals_css__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @/styles/globals.css */ \"./styles/globals.css\");\n/* harmony import */ var _styles_globals_css__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(_styles_globals_css__WEBPACK_IMPORTED_MODULE_3__);\nvar __webpack_async_dependencies__ = __webpack_handle_async_dependencies__([react_hot_toast__WEBPACK_IMPORTED_MODULE_1__]);\nreact_hot_toast__WEBPACK_IMPORTED_MODULE_1__ = (__webpack_async_dependencies__.then ? (await __webpack_async_dependencies__)() : __webpack_async_dependencies__)[0];\n\n\n\n\nfunction App({ Component, pageProps }) {\n    // Suppress MetaMask and other extension errors\n    (0,react__WEBPACK_IMPORTED_MODULE_2__.useEffect)(()=>{\n        const handleError = (event)=>{\n            // Suppress MetaMask and crypto wallet extension errors\n            if (event.message?.includes(\"MetaMask\") || event.message?.includes(\"ethereum\") || event.filename?.includes(\"chrome-extension\") || event.filename?.includes(\"moz-extension\")) {\n                event.preventDefault();\n                event.stopPropagation();\n                return false;\n            }\n        };\n        const handleRejection = (event)=>{\n            // Suppress extension-related promise rejections\n            const reason = event.reason?.toString() || \"\";\n            if (reason.includes(\"MetaMask\") || reason.includes(\"ethereum\") || reason.includes(\"chrome-extension\")) {\n                event.preventDefault();\n                return false;\n            }\n        };\n        window.addEventListener(\"error\", handleError);\n        window.addEventListener(\"unhandledrejection\", handleRejection);\n        return ()=>{\n            window.removeEventListener(\"error\", handleError);\n            window.removeEventListener(\"unhandledrejection\", handleRejection);\n        };\n    }, []);\n    return /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.Fragment, {\n        children: [\n            /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(Component, {\n                ...pageProps\n            }, void 0, false, {\n                fileName: \"/Users/jarvis/Documents/Oviya EI/oviya-website/pages/_app.tsx\",\n                lineNumber: 47,\n                columnNumber: 7\n            }, this),\n            /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(react_hot_toast__WEBPACK_IMPORTED_MODULE_1__.Toaster, {\n                position: \"top-right\",\n                toastOptions: {\n                    duration: 3000,\n                    style: {\n                        background: \"#363636\",\n                        color: \"#fff\"\n                    },\n                    success: {\n                        duration: 3000,\n                        iconTheme: {\n                            primary: \"#10b981\",\n                            secondary: \"#fff\"\n                        }\n                    },\n                    error: {\n                        duration: 4000,\n                        iconTheme: {\n                            primary: \"#ef4444\",\n                            secondary: \"#fff\"\n                        }\n                    }\n                }\n            }, void 0, false, {\n                fileName: \"/Users/jarvis/Documents/Oviya EI/oviya-website/pages/_app.tsx\",\n                lineNumber: 48,\n                columnNumber: 7\n            }, this)\n        ]\n    }, void 0, true);\n}\n\n__webpack_async_result__();\n} catch(e) { __webpack_async_result__(e); } });//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiLi9wYWdlcy9fYXBwLnRzeCIsIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7QUFDeUM7QUFDUjtBQUNKO0FBRWQsU0FBU0UsSUFBSSxFQUFFQyxTQUFTLEVBQUVDLFNBQVMsRUFBWTtJQUM1RCwrQ0FBK0M7SUFDL0NILGdEQUFTQSxDQUFDO1FBQ1IsTUFBTUksY0FBYyxDQUFDQztZQUNuQix1REFBdUQ7WUFDdkQsSUFDRUEsTUFBTUMsT0FBTyxFQUFFQyxTQUFTLGVBQ3hCRixNQUFNQyxPQUFPLEVBQUVDLFNBQVMsZUFDeEJGLE1BQU1HLFFBQVEsRUFBRUQsU0FBUyx1QkFDekJGLE1BQU1HLFFBQVEsRUFBRUQsU0FBUyxrQkFDekI7Z0JBQ0FGLE1BQU1JLGNBQWM7Z0JBQ3BCSixNQUFNSyxlQUFlO2dCQUNyQixPQUFPO1lBQ1Q7UUFDRjtRQUVBLE1BQU1DLGtCQUFrQixDQUFDTjtZQUN2QixnREFBZ0Q7WUFDaEQsTUFBTU8sU0FBU1AsTUFBTU8sTUFBTSxFQUFFQyxjQUFjO1lBQzNDLElBQ0VELE9BQU9MLFFBQVEsQ0FBQyxlQUNoQkssT0FBT0wsUUFBUSxDQUFDLGVBQ2hCSyxPQUFPTCxRQUFRLENBQUMscUJBQ2hCO2dCQUNBRixNQUFNSSxjQUFjO2dCQUNwQixPQUFPO1lBQ1Q7UUFDRjtRQUVBSyxPQUFPQyxnQkFBZ0IsQ0FBQyxTQUFTWDtRQUNqQ1UsT0FBT0MsZ0JBQWdCLENBQUMsc0JBQXNCSjtRQUU5QyxPQUFPO1lBQ0xHLE9BQU9FLG1CQUFtQixDQUFDLFNBQVNaO1lBQ3BDVSxPQUFPRSxtQkFBbUIsQ0FBQyxzQkFBc0JMO1FBQ25EO0lBQ0YsR0FBRyxFQUFFO0lBRUwscUJBQ0U7OzBCQUNFLDhEQUFDVDtnQkFBVyxHQUFHQyxTQUFTOzs7Ozs7MEJBQ3hCLDhEQUFDSixvREFBT0E7Z0JBQ05rQixVQUFTO2dCQUNUQyxjQUFjO29CQUNaQyxVQUFVO29CQUNWQyxPQUFPO3dCQUNMQyxZQUFZO3dCQUNaQyxPQUFPO29CQUNUO29CQUNBQyxTQUFTO3dCQUNQSixVQUFVO3dCQUNWSyxXQUFXOzRCQUNUQyxTQUFTOzRCQUNUQyxXQUFXO3dCQUNiO29CQUNGO29CQUNBQyxPQUFPO3dCQUNMUixVQUFVO3dCQUNWSyxXQUFXOzRCQUNUQyxTQUFTOzRCQUNUQyxXQUFXO3dCQUNiO29CQUNGO2dCQUNGOzs7Ozs7OztBQUlSIiwic291cmNlcyI6WyJ3ZWJwYWNrOi8vb3ZpeWEtd2Vic2l0ZS8uL3BhZ2VzL19hcHAudHN4PzJmYmUiXSwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHR5cGUgeyBBcHBQcm9wcyB9IGZyb20gJ25leHQvYXBwJ1xuaW1wb3J0IHsgVG9hc3RlciB9IGZyb20gJ3JlYWN0LWhvdC10b2FzdCdcbmltcG9ydCB7IHVzZUVmZmVjdCB9IGZyb20gJ3JlYWN0J1xuaW1wb3J0ICdAL3N0eWxlcy9nbG9iYWxzLmNzcydcblxuZXhwb3J0IGRlZmF1bHQgZnVuY3Rpb24gQXBwKHsgQ29tcG9uZW50LCBwYWdlUHJvcHMgfTogQXBwUHJvcHMpIHtcbiAgLy8gU3VwcHJlc3MgTWV0YU1hc2sgYW5kIG90aGVyIGV4dGVuc2lvbiBlcnJvcnNcbiAgdXNlRWZmZWN0KCgpID0+IHtcbiAgICBjb25zdCBoYW5kbGVFcnJvciA9IChldmVudDogRXJyb3JFdmVudCkgPT4ge1xuICAgICAgLy8gU3VwcHJlc3MgTWV0YU1hc2sgYW5kIGNyeXB0byB3YWxsZXQgZXh0ZW5zaW9uIGVycm9yc1xuICAgICAgaWYgKFxuICAgICAgICBldmVudC5tZXNzYWdlPy5pbmNsdWRlcygnTWV0YU1hc2snKSB8fFxuICAgICAgICBldmVudC5tZXNzYWdlPy5pbmNsdWRlcygnZXRoZXJldW0nKSB8fFxuICAgICAgICBldmVudC5maWxlbmFtZT8uaW5jbHVkZXMoJ2Nocm9tZS1leHRlbnNpb24nKSB8fFxuICAgICAgICBldmVudC5maWxlbmFtZT8uaW5jbHVkZXMoJ21vei1leHRlbnNpb24nKVxuICAgICAgKSB7XG4gICAgICAgIGV2ZW50LnByZXZlbnREZWZhdWx0KClcbiAgICAgICAgZXZlbnQuc3RvcFByb3BhZ2F0aW9uKClcbiAgICAgICAgcmV0dXJuIGZhbHNlXG4gICAgICB9XG4gICAgfVxuXG4gICAgY29uc3QgaGFuZGxlUmVqZWN0aW9uID0gKGV2ZW50OiBQcm9taXNlUmVqZWN0aW9uRXZlbnQpID0+IHtcbiAgICAgIC8vIFN1cHByZXNzIGV4dGVuc2lvbi1yZWxhdGVkIHByb21pc2UgcmVqZWN0aW9uc1xuICAgICAgY29uc3QgcmVhc29uID0gZXZlbnQucmVhc29uPy50b1N0cmluZygpIHx8ICcnXG4gICAgICBpZiAoXG4gICAgICAgIHJlYXNvbi5pbmNsdWRlcygnTWV0YU1hc2snKSB8fFxuICAgICAgICByZWFzb24uaW5jbHVkZXMoJ2V0aGVyZXVtJykgfHxcbiAgICAgICAgcmVhc29uLmluY2x1ZGVzKCdjaHJvbWUtZXh0ZW5zaW9uJylcbiAgICAgICkge1xuICAgICAgICBldmVudC5wcmV2ZW50RGVmYXVsdCgpXG4gICAgICAgIHJldHVybiBmYWxzZVxuICAgICAgfVxuICAgIH1cblxuICAgIHdpbmRvdy5hZGRFdmVudExpc3RlbmVyKCdlcnJvcicsIGhhbmRsZUVycm9yKVxuICAgIHdpbmRvdy5hZGRFdmVudExpc3RlbmVyKCd1bmhhbmRsZWRyZWplY3Rpb24nLCBoYW5kbGVSZWplY3Rpb24pXG5cbiAgICByZXR1cm4gKCkgPT4ge1xuICAgICAgd2luZG93LnJlbW92ZUV2ZW50TGlzdGVuZXIoJ2Vycm9yJywgaGFuZGxlRXJyb3IpXG4gICAgICB3aW5kb3cucmVtb3ZlRXZlbnRMaXN0ZW5lcigndW5oYW5kbGVkcmVqZWN0aW9uJywgaGFuZGxlUmVqZWN0aW9uKVxuICAgIH1cbiAgfSwgW10pXG5cbiAgcmV0dXJuIChcbiAgICA8PlxuICAgICAgPENvbXBvbmVudCB7Li4ucGFnZVByb3BzfSAvPlxuICAgICAgPFRvYXN0ZXIgXG4gICAgICAgIHBvc2l0aW9uPVwidG9wLXJpZ2h0XCJcbiAgICAgICAgdG9hc3RPcHRpb25zPXt7XG4gICAgICAgICAgZHVyYXRpb246IDMwMDAsXG4gICAgICAgICAgc3R5bGU6IHtcbiAgICAgICAgICAgIGJhY2tncm91bmQ6ICcjMzYzNjM2JyxcbiAgICAgICAgICAgIGNvbG9yOiAnI2ZmZicsXG4gICAgICAgICAgfSxcbiAgICAgICAgICBzdWNjZXNzOiB7XG4gICAgICAgICAgICBkdXJhdGlvbjogMzAwMCxcbiAgICAgICAgICAgIGljb25UaGVtZToge1xuICAgICAgICAgICAgICBwcmltYXJ5OiAnIzEwYjk4MScsXG4gICAgICAgICAgICAgIHNlY29uZGFyeTogJyNmZmYnLFxuICAgICAgICAgICAgfSxcbiAgICAgICAgICB9LFxuICAgICAgICAgIGVycm9yOiB7XG4gICAgICAgICAgICBkdXJhdGlvbjogNDAwMCxcbiAgICAgICAgICAgIGljb25UaGVtZToge1xuICAgICAgICAgICAgICBwcmltYXJ5OiAnI2VmNDQ0NCcsXG4gICAgICAgICAgICAgIHNlY29uZGFyeTogJyNmZmYnLFxuICAgICAgICAgICAgfSxcbiAgICAgICAgICB9LFxuICAgICAgICB9fVxuICAgICAgLz5cbiAgICA8Lz5cbiAgKVxufSJdLCJuYW1lcyI6WyJUb2FzdGVyIiwidXNlRWZmZWN0IiwiQXBwIiwiQ29tcG9uZW50IiwicGFnZVByb3BzIiwiaGFuZGxlRXJyb3IiLCJldmVudCIsIm1lc3NhZ2UiLCJpbmNsdWRlcyIsImZpbGVuYW1lIiwicHJldmVudERlZmF1bHQiLCJzdG9wUHJvcGFnYXRpb24iLCJoYW5kbGVSZWplY3Rpb24iLCJyZWFzb24iLCJ0b1N0cmluZyIsIndpbmRvdyIsImFkZEV2ZW50TGlzdGVuZXIiLCJyZW1vdmVFdmVudExpc3RlbmVyIiwicG9zaXRpb24iLCJ0b2FzdE9wdGlvbnMiLCJkdXJhdGlvbiIsInN0eWxlIiwiYmFja2dyb3VuZCIsImNvbG9yIiwic3VjY2VzcyIsImljb25UaGVtZSIsInByaW1hcnkiLCJzZWNvbmRhcnkiLCJlcnJvciJdLCJzb3VyY2VSb290IjoiIn0=\n//# sourceURL=webpack-internal:///./pages/_app.tsx\n");

/***/ }),

/***/ "./styles/globals.css":
/*!****************************!*\
  !*** ./styles/globals.css ***!
  \****************************/
/***/ (() => {



/***/ }),

/***/ "react":
/*!************************!*\
  !*** external "react" ***!
  \************************/
/***/ ((module) => {

"use strict";
module.exports = require("react");

/***/ }),

/***/ "react/jsx-dev-runtime":
/*!****************************************!*\
  !*** external "react/jsx-dev-runtime" ***!
  \****************************************/
/***/ ((module) => {

"use strict";
module.exports = require("react/jsx-dev-runtime");

/***/ }),

/***/ "react-hot-toast":
/*!**********************************!*\
  !*** external "react-hot-toast" ***!
  \**********************************/
/***/ ((module) => {

"use strict";
module.exports = import("react-hot-toast");;

/***/ })

};
;

// load runtime
var __webpack_require__ = require("../webpack-runtime.js");
__webpack_require__.C(exports);
var __webpack_exec__ = (moduleId) => (__webpack_require__(__webpack_require__.s = moduleId))
var __webpack_exports__ = (__webpack_exec__("./pages/_app.tsx"));
module.exports = __webpack_exports__;

})();