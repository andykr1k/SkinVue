export default function LandingPage() {
  return (
    <div className="bg-ghostwhite dark:bg-richblack p-5 dark:text-ghostwhite text-richblack ">
      <div className="flex justify-center">
        <img
          className="w-48 hidden dark:block"
          src="/WhiteLogo.webp"
          alt="WebP Image"
        />
        <img
          className="w-48 dark:hidden"
          src="/BlackLogoNoText.webp"
          alt="WebP Image"
        />{" "}
      </div>
      <div className="flex justify-center mx-3 text-3xl font-bold sm:text-4xl">
        <h1>SkinVue</h1>
      </div>
      <section className="mt-2">
        <div className="px-4 py-4 sm:px-6 sm:py-8 lg:px-8 ">
          <div className="grid grid-cols-1 gap-y-8">
            <div className="flex justify-center">
              <div className="mx-auto max-w-lg text-center lg:mx-0 ltr:lg:text-left rtl:lg:text-right">
                <h2 className="text-2xl font-bold sm:text-3xl dark:text-ghostwhite text-richblack">
                  Analyze and Track Skin Lesions
                </h2>

                <p className="mt-4 text-gray-600">
                  SkinVue is a personalized user-friendly device that detects
                  potentially cancerous moles. It utilizes convultional nueral
                  networks and user information for better accuracy.
                </p>

                <a
                  href="/dashboard"
                  className="mt-8 inline-block rounded bg-indigo-600 px-12 py-3 text-sm font-medium text-white transition hover:bg-indigo-700 focus:outline-none focus:ring focus:ring-yellow-400"
                >
                  Get Started
                </a>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 mt-4">
              <div className="block rounded-xl p-4 shadow-sm focus:outline-none focus:ring bg-black/10 dark:text-ghostwhite text-richblack">
                <h2 className="mt-2 font-bold">Scan</h2>

                <p className="hidden sm:mt-1 sm:block sm:text-sm sm:text-gray-600">
                  Easy way to track and take a picture with a SkinVue device.
                </p>
              </div>

              <div className="block rounded-xl p-4 shadow-sm focus:outline-none focus:ring bg-black/10 dark:text-ghostwhite text-richblack">
                <h2 className="mt-2 font-bold">Analyze</h2>

                <p className="hidden sm:mt-1 sm:block sm:text-sm sm:text-gray-600">
                  Here you will find your personalized description of the most
                  recent scan.
                </p>
              </div>

              <div className="block rounded-xl p-4 shadow-sm focus:outline-none focus:ring bg-black/10 dark:text-ghostwhite text-richblack">
                <h2 className="mt-2 font-bold">Track</h2>

                <p className="hidden sm:mt-1 sm:block sm:text-sm sm:text-gray-600">
                  Compare the latest analysis with the previous descriptions of
                  the same melanoma location.
                </p>
              </div>

              <div className="block rounded-xl p-4 shadow-sm focus:outline-none focus:ring bg-black/10 dark:text-ghostwhite text-richblack">
                <h2 className="mt-2 font-bold">History</h2>

                <p className="hidden sm:mt-1 sm:block sm:text-sm sm:text-gray-600">
                  Store reports of all the classifications made by SkinVue to
                  show your doctor.
                </p>
              </div>

              <div className="block rounded-xl p-4 shadow-sm focus:outline-none focus:ring bg-black/10 dark:text-ghostwhite text-richblack">
                <h2 className="mt-2 font-bold">Find A Doc</h2>

                <p className="hidden sm:mt-1 sm:block sm:text-sm sm:text-gray-600">
                  Find the nearest dermatologist, for any questions or concerns
                  about your analysis.
                </p>
              </div>

              <div className="block rounded-xl p-4 shadow-sm focus:outline-none focus:ring bg-black/10 dark:text-ghostwhite text-richblack">
                <h2 className="mt-2 font-bold">About Us</h2>

                <p className="hidden sm:mt-1 sm:block sm:text-sm sm:text-gray-600">
                  Discover what inspired us to create this wearable health
                  device and interface. Baer hack 2024 UCR
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
      <footer className="bg-ghostwhite dark:bg-richblack dark:text-ghostwhite text-richblack">
        <div className="mx-auto max-w-screen-xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="mt-8">
            <div className="mt-8 sm:flex sm:items-center sm:justify-between">
              <div className="flex justify-center text-teal-600 sm:justify-start">
                <ul className="flex justify-center gap-6">
                  <li>
                    <a
                      href="https://www.github.com/andykr1k/skinvue"
                      rel="noreferrer"
                      target="_blank"
                      className="text-teal-700 transition hover:text-teal-700/75"
                    >
                      <span className="sr-only">GitHub</span>
                      <svg
                        className="h-6 w-6"
                        fill="currentColor"
                        viewBox="0 0 24 24"
                        aria-hidden="true"
                      >
                        <path
                          fillRule="evenodd"
                          d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </a>
                  </li>

                </ul>
              </div>

              <p className="mt-4 text-center text-sm text-gray-500 sm:mt-0 sm:text-right">
                Copyright &copy; 2024. All rights reserved.
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
