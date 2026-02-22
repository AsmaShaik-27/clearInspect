export default function Header() {
  return (
    <header className="bg-black text-white py-4 shadow-md">
      <div className="container mx-auto flex justify-between px-6">
        <h1 className="text-2xl font-bold text-teal-400">
          ClearInspect 
        </h1>
        <nav>
          <a href="#about" className="mx-4 hover:text-teal-400">
            About
          </a>
          <a href="#demo" className="mx-4 hover:text-teal-400">
            Demo
          </a>
        </nav>
      </div>
    </header>
  );
}