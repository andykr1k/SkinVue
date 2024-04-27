export default function LogInPage() {
  return (
    <div className="flex h-screen w-full items-center justify-center bg-ghostwhite px-4 dark:bg-richblack">
      <div className="w-full max-w-md space-y-6">
        <div className="text-center">
          <h1 className="text-3xl font-bold tracking-tight">SkinVue</h1>
          <p className="text-gray-500 dark:text-gray-400">
            Enter your credentials to access your account.
          </p>
        </div>
        <form className="space-y-4">
          <div className="flex justify-around">
            <input
              className="p-2 rounded-md w-full"
              id="username"
              placeholder="Enter your username"
              required
              type="text"
            />
          </div>
          <div className="flex justify-around">
            <input
              className="p-2 rounded-md w-full"
              id="password"
              placeholder="Enter your password"
              required
              type="password"
            />
          </div>
          <button className="w-full bg-slate-700 p-2 rounded-md" type="submit">
            Sign in
          </button>
        </form>
      </div>
    </div>
  );
}