export default function Dashboard() {
    return (
      <div className="p-5 lg:h-screen bg-ghostwhite dark:bg-richblack">
        <div className="flex justify-between mx-3 dark:text-ghostwhite text-richblack">
          <div>
            <h1>SkinVue</h1>
          </div>
          <div>
            <h2>Username</h2>
          </div>
        </div>
        <div className="hidden lg:flex justify-center space-x-3 mt-5">
          <div className="bg-dimgrey w-[768px] h-[768px] rounded-md"></div>
          <div className="space-y-3">
            <div className="bg-dimgrey w-48 h-[248px] rounded-md"></div>
            <div className="bg-dimgrey w-48 h-[248px] rounded-md"></div>
            <div className="bg-dimgrey w-48 h-[248px] rounded-md"></div>
          </div>
          <div className="space-y-3">
            <div className="bg-dimgrey w-[396px] h-[508px] rounded-md"></div>
            <div className="flex space-x-3">
              <div className="bg-dimgrey w-48 h-[248px] rounded-md"></div>
              <div className="bg-dimgrey w-48 h-[248px] rounded-md"></div>
            </div>
          </div>
        </div>
        <div className="grid lg:hidden mt-5 space-y-3">
          <div className="bg-dimgrey w-full h-[768px] rounded-md"></div>
          <div className="flex space-x-3">
            <div className="bg-dimgrey w-1/2 h-[248px] rounded-md"></div>
            <div className="bg-dimgrey w-1/2 h-[248px] rounded-md"></div>
          </div>
          <div className="space-y-3">
            <div className="bg-dimgrey w-full h-[508px] rounded-md"></div>
            <div className="flex space-x-3">
              <div className="bg-dimgrey w-1/2 h-[248px] rounded-md"></div>
              <div className="bg-dimgrey w-1/2 h-[248px] rounded-md"></div>
            </div>
            <div className="flex space-x-3">
              <div className="bg-dimgrey w-1/2 h-[248px] rounded-md"></div>
            </div>
          </div>
        </div>
      </div>
    );
}